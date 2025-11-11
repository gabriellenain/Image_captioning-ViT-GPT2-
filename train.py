import os, time, csv, argparse, math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup
from torch import amp

from data import FeatCaptionDataset
from model import CaptionGPT2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@torch.no_grad()
def evaluate(model, dl, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True, help="train_feats.jsonl (feat_path/shard + caption)")
    ap.add_argument("--val_jsonl", default=None, help="val_feats.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--samples_every", type=int, default=0)
    ap.add_argument("--gpt", type=str, default="gpt2")
    ap.add_argument("--custom_ckpt", type=str, default=None)
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--log_every", type=int, default=100, help="flush metrics to disk every N optimizer steps")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "train_metrics.csv")
    ckpt_last   = os.path.join(args.out_dir, "cap_xattn_last.pt")
    ckpt_best   = os.path.join(args.out_dir, "cap_xattn_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = GPT2TokenizerFast.from_pretrained(args.gpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = FeatCaptionDataset(args.train_jsonl, tokenizer=tok, max_len=48)
    val_ds   = FeatCaptionDataset(args.val_jsonl, tokenizer=tok, max_len=48) if args.val_jsonl else None

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

    model = CaptionGPT2(
        enc_dim=enc_dim,
        tokenizer=tok,
        gpt_name=args.gpt,
        freeze_lm=True,
        custom_ckpt=args.custom_ckpt
    ).to(device)
    model.train()

    params = (p for p in model.parameters() if p.requires_grad)
    try:
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    except TypeError:
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    scaler = amp.GradScaler("cuda", enabled=(args.fp16 and device == "cuda"))

    total_opt_steps = max(1, args.epochs * math.ceil(len(train_ds) / (args.batch_size * max(1, args.grad_accum))))
    warmup = max(10, total_opt_steps // 20)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup, num_training_steps=total_opt_steps
    )

    best_val = float("inf")
    opt_step = 0

    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss", "val_loss", "time"])

    metrics_buf = []

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        optim.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar, start=1):
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

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                metrics_buf.append([epoch, opt_step, f"{loss.item():.6f}", "", str(int(time.time()))])

                if len(metrics_buf) >= args.log_every:
                    with open(metrics_csv, "a", newline="") as f:
                        csv.writer(f).writerows(metrics_buf)
                    metrics_buf.clear()

            if args.samples_every and opt_step > 0 and (opt_step % args.samples_every == 0) and (i % args.grad_accum == 0):
                model.eval()
                with torch.no_grad():
                    sub_pt = pt[:4]
                    prompt_ids = tok(["A photo of"] * sub_pt.size(0), return_tensors="pt", padding=True).input_ids.to(device)
                    texts = model.generate(sub_pt, prompt_ids, max_new_tokens=24, temperature=0.7, top_p=0.9)
                    for t in texts:
                        print("[sample]", t)
                model.train()

        if metrics_buf:
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerows(metrics_buf)
            metrics_buf.clear()

        val_loss = evaluate(model, val_dl, device) if val_dl else None
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, opt_step, "", f"{val_loss:.6f}" if val_loss is not None else "", str(int(time.time()))])

        torch.save({
            "model_state": model.state_dict(),
            "gpt_name": args.gpt,
            "enc_dim": enc_dim,
            "epoch": epoch,
            "step": opt_step,
            "val_loss": val_loss,
        }, ckpt_last)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "gpt_name": args.gpt,
                "enc_dim": enc_dim,
                "epoch": epoch,
                "step": opt_step,
                "val_loss": val_loss,
            }, ckpt_best)

        print(f"[epoch {epoch}] val_loss={val_loss:.4f}" if val_loss is not None else f"[epoch {epoch}]")

    print(f"Saved last: {ckpt_last}")
    if os.path.exists(ckpt_best):
        print(f"Best: {ckpt_best}")


if __name__ == "__main__":
    main()
