import os, time, csv, argparse, math

os.environ["HF_HOME"] = "/Data/gabriel.lenain/hf_cache"
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]

from torch.profiler import profile, ProfilerActivity
import tiktoken
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, CLIPModel, CLIPTokenizer
from torch import amp

from data import FeatCaptionDataset
from model import CaptionNanoGPT

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    Cider = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


class TiktokenTokenizer:
    """
    Minimal wrapper so the rest of the code can use:
      - tokenizer(..., return_tensors="pt", padding=True, max_length, truncation)
      - tokenizer.eos_token_id, tokenizer.pad_token_id
      - tokenizer.batch_decode(...)
    """
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.eos_token_id = self.enc.eot_token
        self.pad_token_id = self.eos_token_id  

    def __call__(self, texts, return_tensors=None,
                 padding=False, max_length=None, truncation=False):
        if isinstance(texts, str):
            texts = [texts]

        all_ids = []
        for t in texts:
            ids = self.enc.encode(t)
            ids.append(self.eos_token_id)
            if max_length is not None and truncation and len(ids) > max_length:
                ids = ids[:max_length]
            all_ids.append(ids)

        if padding:
            if max_length is not None:
                tgt_len = max_length
            else:
                tgt_len = max(len(x) for x in all_ids)
            padded_ids = []
            padded_mask = []
            for ids in all_ids:
                if len(ids) < tgt_len:
                    pad_len = tgt_len - len(ids)
                    padded_ids.append(ids + [self.pad_token_id] * pad_len)
                    padded_mask.append([1] * len(ids) + [0] * pad_len)
                else:
                    ids = ids[:tgt_len]
                    padded_ids.append(ids)
                    padded_mask.append([1] * len(ids))
            all_ids = padded_ids
        else:
            padded_mask = [[1] * len(ids) for ids in all_ids]

        if return_tensors == "pt":
            input_ids = torch.tensor(all_ids, dtype=torch.long)
            attention_mask = torch.tensor(padded_mask, dtype=torch.long)

            class Batch:
                pass
            b = Batch()
            b.input_ids = input_ids
            b.attention_mask = attention_mask
            return b

        raise NotImplementedError("This tokenizer is only implemented for return_tensors='pt'.")

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        texts = []
        for ids in batch_ids:
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if skip_special_tokens and self.eos_token_id in ids:
                idx = ids.index(self.eos_token_id)
                ids = ids[:idx]
            texts.append(self.enc.decode(ids))
        return texts


@torch.no_grad()
def evaluate(model, dl, device):
    """Standard validation loss on full val set."""
    if dl is None:
        return None
    model.eval()
    tot, n = 0.0, 0
    for batch in dl:
        pt   = batch["patch_tokens"].to(device, non_blocking=True).float()
        ids  = batch["input_ids"].to(device, non_blocking=True).long()
        lbls = batch["labels"].to(device, non_blocking=True).long()
        _, loss = model(pt, ids, labels=lbls)
        bsz = pt.size(0)
        tot += loss.item() * bsz
        n   += bsz
    model.train()
    return tot / max(1, n)


@torch.no_grad()
def evaluate_cider(model, dl, tokenizer, device,
                   max_batches=50, max_new_tokens=24):
    """
    Approximate COCO CIDEr on a subset of val.
    Requires pycocoevalcap. Returns None if unavailable.
    Assumes each batch has a 'caption' field with reference strings.
    """
    if dl is None or Cider is None:
        return None

    model.eval()
    scorer = Cider()

    gts, res = {}, {}
    img_id = 0

    for b_idx, batch in enumerate(dl):
        if b_idx >= max_batches:
            break

        pt = batch["patch_tokens"].to(device, non_blocking=True).float()
        refs = batch["caption"]

        prompt_ids = tokenizer(
            ["A photo of"] * pt.size(0),
            return_tensors="pt",
            padding=True,
            max_length=16,
            truncation=True,
        ).input_ids.to(device)

        preds = model.generate(
            pt,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        for r, p in zip(refs, preds):
            gts[img_id] = [r]
            res[img_id] = [p]
            img_id += 1

    score, _ = scorer.compute_score(gts, res)
    model.train()
    return float(score)


@torch.no_grad()
def evaluate_clipscore(model, dl, device, tokenizer,
                       max_batches=50, max_new_tokens=24):
    """
    Text-only CLIP-based similarity between reference captions and
    generated captions, using 'openai/clip-vit-large-patch14' text encoder.

    This is not the canonical image–text CLIPScore, but a semantic
    similarity metric in CLIP text space.
    """
    if dl is None:
        return None

    if not hasattr(evaluate_clipscore, "_clip"):
        clip_name = "openai/clip-vit-large-patch14"
        clip_model = CLIPModel.from_pretrained(clip_name).to(device)
        clip_model.eval()
        clip_tok = CLIPTokenizer.from_pretrained(clip_name)
        evaluate_clipscore._clip = (clip_model, clip_tok)

    clip_model, clip_tok = evaluate_clipscore._clip

    model.eval()
    sims = []

    for b_idx, batch in enumerate(dl):
        if b_idx >= max_batches:
            break

        pt = batch["patch_tokens"].to(device, non_blocking=True).float()
        refs = batch["caption"]  

        prompt_ids = tokenizer(
            ["A photo of"] * pt.size(0),
            return_tensors="pt",
            padding=True,
            max_length=16,
            truncation=True,
        ).input_ids.to(device)

        preds = model.generate(
            pt,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

        inputs_ref = clip_tok(
            list(refs),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        inputs_pred = clip_tok(
            list(preds),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        emb_ref = clip_model.get_text_features(**inputs_ref)   # (B, D)
        emb_pred = clip_model.get_text_features(**inputs_pred) # (B, D)

        # cosine similarity
        emb_ref = emb_ref / (emb_ref.norm(dim=-1, keepdim=True) + 1e-8)
        emb_pred = emb_pred / (emb_pred.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (emb_ref * emb_pred).sum(dim=-1)  # (B,)

        sims.append(cos.detach().cpu())

    model.train()

    if not sims:
        return None

    all_sims = torch.cat(sims, dim=0)
    return float(all_sims.mean())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train_jsonl",
        default="/Data/gabriel.lenain/image-captioner/datasets/coco2017_full/train_feats_clip.jsonl",
        help="train_feats.jsonl (feat_path/shard + caption)",
    )
    ap.add_argument(
        "--val_jsonl",
        default="/Data/gabriel.lenain/image-captioner/datasets/coco2017_full/val_feats_clip.jsonl",
        help="val_feats.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        default="/Data/gabriel.lenain/image-captioner/runs/Linear_Model",
        help="output directory for checkpoints + logs",
    )

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--samples_every", type=int, default=0)

    ap.add_argument("--gpt", type=str, default="gpt2")

    ap.add_argument(
        "--custom_ckpt",
        type=str,
        default="/users/eleves-b/2022/gabriel.lenain/ZerotoHero/log/model_19073_flat.pt",
    )

    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--log_every", type=int, default=100, help="flush metrics to disk every N optimizer steps")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="stop after this many optimizer steps (if set)",
    )

    ap.add_argument(
        "--m_vis_tokens",
        type=int,
        default=32,
        help="number of visual tokens used as prefix (None = use all)",
    )

    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=200,
        help="number of validation evaluations without improvement before stopping (<=0 to disable)",
    )

    args = ap.parse_args()
    args.fp16 = True

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "train_metrics.csv")
    ckpt_last   = os.path.join(args.out_dir, "cap_xattn_last.pt")
    ckpt_best   = os.path.join(args.out_dir, "cap_xattn_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = TiktokenTokenizer("gpt2")

    train_ds = FeatCaptionDataset(
        args.train_jsonl,
        tokenizer=tok,
        max_len=32,
    )
    val_ds   = FeatCaptionDataset(
        args.val_jsonl,
        tokenizer=tok,
        max_len=32,
    ) if args.val_jsonl else None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    val_dl = None
    if val_ds:
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, args.num_workers),
            pin_memory=(device == "cuda"),
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=(args.num_workers > 0),
            drop_last=False,
        )

    enc_dim = int(train_ds[0]["patch_tokens"].shape[-1])

    model = CaptionNanoGPT(
        enc_dim=enc_dim,
        tokenizer=tok,
        freeze_lm=True,
        custom_ckpt=args.custom_ckpt,
        m_vis_tokens=args.m_vis_tokens,
        use_cls_only=False,
    ).to(device)

    if FlopCountAnalysis is not None:
        try:
            b0 = next(iter(train_dl))
            pt0  = b0["patch_tokens"].to(device).float()
            ids0 = b0["input_ids"].to(device).long()

            fca = FlopCountAnalysis(model, (pt0, ids0))
            total_flops = fca.total()
            print(f"[FLOPs] ~{total_flops / 1e9:.2f} GFLOPs per forward pass (batch={pt0.size(0)})")

            print("\n[Top modules by FLOPs]")
            by_module = fca.by_module()
            for name, flops in sorted(by_module.items(), key=lambda x: x[1], reverse=True)[:15]:
                print(f"{name:45s} {flops / 1e9:8.3f} GFLOPs")

            print("\n[FLOPs by operator]")
            by_op = fca.by_operator()
            for op, flops in sorted(by_op.items(), key=lambda x: x[1], reverse=True):
                print(f"{op:25s} {flops / 1e9:8.3f} GFLOPs")

        except Exception as e:
            print(f"[FLOPs] unable to compute FLOPs: {e}")
    else:
        print("[FLOPs] fvcore not installed, skipping FLOP counting")

    params = (p for p in model.parameters() if p.requires_grad)
    try:
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    except TypeError:
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    scaler = amp.GradScaler("cuda", enabled=(args.fp16 and device == "cuda"))

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * max(1, args.grad_accum)))

    if args.max_steps is not None:
        total_opt_steps = args.max_steps
        args.epochs = max(1, math.ceil(args.max_steps / steps_per_epoch))
    else:
        total_opt_steps = max(
            1,
            args.epochs * steps_per_epoch,
        )

    warmup = max(10, total_opt_steps // 20)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup, num_training_steps=total_opt_steps
    )

    eval_interval = 50

    best_val = float("inf")
    opt_step = 0
    no_improve = 0
    stop_training = False

    PROFILE = True if device == "cuda" else False
    PROFILE_MAX_STEPS = 20
    prof = None
    if PROFILE:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        )
        prof.__enter__()

    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "step", "loss", "val_loss", "val_cider", "val_clip", "time"]
        )

    metrics_buf = []

    def run_validation(epoch, step):
        nonlocal best_val, no_improve, stop_training

        val_loss = evaluate(model, val_dl, device) if val_dl else None
        val_cider = evaluate_cider(model, val_dl, tok, device) if val_dl else None
        val_clip = evaluate_clipscore(model, val_dl, device, tok) if val_dl else None

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                step,
                "",
                f"{val_loss:.6f}" if val_loss is not None else "",
                f"{val_cider:.4f}" if val_cider is not None else "",
                f"{val_clip:.4f}" if val_clip is not None else "",
                str(int(time.time())),
            ])

        if val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "gpt_name": args.gpt,
                    "enc_dim": enc_dim,
                    "epoch": epoch,
                    "step": step,
                    "val_loss": val_loss,
                    "val_cider": val_cider,
                    "val_clip": val_clip,
                }, ckpt_best)
                print(f"[val] step={step} NEW BEST val_loss={val_loss:.4f}, "
                      f"CIDEr={val_cider}, CLIP={val_clip}")
            else:
                no_improve += 1
                print(f"[val] step={step} val_loss={val_loss:.4f}, "
                      f"CIDEr={val_cider}, CLIP={val_clip}, "
                      f"no_improve={no_improve}")

            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(f"[early stop] no val_loss improvement for {no_improve} evaluations.")
                stop_training = True

        return val_loss, val_cider

    if val_dl:
        run_validation(epoch=0, step=0)

    # ======================= TRAIN LOOP ======================= #
    for epoch in range(1, args.epochs + 1):
        if stop_training:
            break

        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        optim.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar, start=1):
            if stop_training:
                break

            pt   = batch["patch_tokens"].to(device, non_blocking=True).float()
            ids  = batch["input_ids"].to(device, non_blocking=True).long()
            lbls = batch["labels"].to(device, non_blocking=True).long()

            with amp.autocast(device_type="cuda", enabled=(args.fp16 and device == "cuda")):
                _, loss = model(pt, ids, labels=lbls)
                loss_to_backprop = loss / max(1, args.grad_accum)

            scaler.scale(loss_to_backprop).backward()

            if i % max(1, args.grad_accum) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

                sched.step()
                opt_step += 1

                if prof is not None:
                    prof.step()
                    if opt_step >= PROFILE_MAX_STEPS:
                        prof.__exit__(None, None, None)
                        print(prof.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=30,
                        ))
                        prof = None

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                metrics_buf.append([
                    epoch,
                    opt_step,
                    f"{loss.item():.6f}",
                    "",
                    "",
                    "",
                    str(int(time.time())),
                ])

                if len(metrics_buf) >= args.log_every:
                    with open(metrics_csv, "a", newline="") as f:
                        csv.writer(f).writerows(metrics_buf)
                    metrics_buf.clear()

                if args.max_steps is not None and opt_step >= args.max_steps:
                    print(f"[train] Reached max_steps={args.max_steps}, stopping training.")
                    stop_training = True
                    break

                if val_dl and (opt_step % eval_interval == 0):
                    run_validation(epoch, opt_step)

            if (not stop_training and args.samples_every
                and opt_step > 0
                and (opt_step % args.samples_every == 0)
                and (i % args.grad_accum == 0)):
                model.eval()
                with torch.no_grad():
                    sub_pt = pt[:4]
                    prompt_ids = tok(
                        ["A photo of"] * sub_pt.size(0),
                        return_tensors="pt",
                        padding=True,
                        max_length=16,
                        truncation=True,
                    ).input_ids.to(device)
                    texts = model.generate(
                        sub_pt,
                        prompt_ids,
                        max_new_tokens=24,
                        temperature=0.7,
                        top_p=0.9
                    )
                    for t in texts:
                        print("[sample]", t)
                model.train()

        if stop_training:
            break

        if metrics_buf:
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerows(metrics_buf)
            metrics_buf.clear()

        if val_dl:
            run_validation(epoch, opt_step)

        torch.save({
            "model_state": model.state_dict(),
            "gpt_name": args.gpt,
            "enc_dim": enc_dim,
            "epoch": epoch,
            "step": opt_step,
            "val_loss": best_val,
        }, ckpt_last)

    if prof is not None:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        ))

    print(f"Saved last: {ckpt_last}")
    if os.path.exists(ckpt_best):
        print(f"Best: {ckpt_best}")


if __name__ == "__main__":
    main()
