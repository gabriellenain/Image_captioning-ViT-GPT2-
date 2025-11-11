import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
import re

class GPT2SelfAttention(nn.Module):
    def __init__(self, d, n_heads, drop=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))   


class GPT2CrossAttention(nn.Module):
    def __init__(self, d, n_heads, drop=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, txt, img):
        B, T, C = txt.shape
        q = self.q(txt).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(img).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.v(img).view(B, -1, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.o(y))    


class GPT2MLP(nn.Module):
    def __init__(self, d, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.fc = nn.Linear(d, mlp_ratio * d)
        self.proj = nn.Linear(mlp_ratio * d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))  


class VisionBlock(nn.Module):
    """Self-attn → Cross-attn → MLP with pre-LN and residuals (GPT-2 style order)."""
    def __init__(self, d, n_heads, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = GPT2SelfAttention(d, n_heads, drop)
        self.ln2 = nn.LayerNorm(d)
        self.ca = GPT2CrossAttention(d, n_heads, drop)
        self.ln3 = nn.LayerNorm(d)
        self.mlp = GPT2MLP(d, drop=drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x, img_mem):
        x = x + self.drop(self.sa(self.ln1(x)))
        x = x + self.drop(self.ca(self.ln2(x), img_mem))
        x = x + self.drop(self.mlp(self.ln3(x)))
        return x
        
def _safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: 
        return torch.load(path, map_location="cpu")

def _flat_sd(obj):
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "state_dict"):
        obj = obj["model"].state_dict()
    elif not isinstance(obj, dict):
        raise RuntimeError("Unsupported checkpoint object")
    return {k.replace("module.", ""): v for k, v in obj.items()}

def _get_any(sd, names):
    for n in names:
        if n in sd:
            return sd[n]
    for k in sd:
        if any(k.endswith(n) for n in names):
            return sd[k]
    ks = list(sd.keys())
    raise KeyError(f"None of {names} found. Sample keys: {ks[:12]} ... (total {len(ks)})")

def _infer_cfg(sd):
    wte = _get_any(sd, ("transformer.wte.weight", "wte.weight"))
    wpe = _get_any(sd, ("transformer.wpe.weight", "wpe.weight"))
    V, d = wte.shape
    block_size = wpe.shape[0]

    L = 0
    pat = re.compile(r"(?:^|.*\.)h\.(\d+)\.")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            L = max(L, int(m.group(1)) + 1)

    H = 12  
    return dict(vocab_size=V, n_embd=d, n_layer=L, n_head=H, block_size=block_size)

def _copy_linear(dst, src):
    with torch.no_grad():
        if src.weight.shape == dst.weight.shape: dst.weight.copy_(src.weight)
        elif src.weight.T.shape == dst.weight.shape: dst.weight.copy_(src.weight.T)
        else: raise RuntimeError("shape mismatch in linear copy")
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)

class CaptionGPT2(nn.Module):
    def __init__(self, enc_dim, tokenizer, gpt_name="gpt2", freeze_lm=True, custom_ckpt=None):
        super().__init__()
        self.tokenizer = tokenizer

        if custom_ckpt is None:
            base = GPT2LMHeadModel.from_pretrained(gpt_name)
            cfg = base.config
            d, L, H, V = cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.vocab_size
            self.block_size = cfg.n_positions
        else:
            sd = _flat_sd(_safe_load(custom_ckpt))
            cfg = _infer_cfg(sd)
            d, L, H, V = cfg["n_embd"], cfg["n_layer"], cfg["n_head"], cfg["vocab_size"]
            self.block_size = cfg["block_size"]

        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(self.block_size, d)
        self.blocks = nn.ModuleList([VisionBlock(d, H) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.wte.weight 
        self.img_proj = nn.Linear(enc_dim, d)

        with torch.no_grad():
            if custom_ckpt is None:
                base = GPT2LMHeadModel.from_pretrained(gpt_name)
                self.wte.weight.copy_(base.transformer.wte.weight)
                self.wpe.weight.copy_(base.transformer.wpe.weight)
                self.ln_f.weight.copy_(base.transformer.ln_f.weight)
                self.ln_f.bias.copy_(base.transformer.ln_f.bias)
                for i, blk in enumerate(self.blocks):
                    ref = base.transformer.h[i]
                    blk.ln1.weight.copy_(ref.ln_1.weight); blk.ln1.bias.copy_(ref.ln_1.bias)
                    blk.ln3.weight.copy_(ref.ln_2.weight); blk.ln3.bias.copy_(ref.ln_2.bias)
                    _copy_linear(blk.sa.c_attn, ref.attn.c_attn); _copy_linear(blk.sa.c_proj, ref.attn.c_proj)
                    _copy_linear(blk.mlp.fc, ref.mlp.c_fc);      _copy_linear(blk.mlp.proj, ref.mlp.c_proj)
            else:
                sd = _flat_sd(_safe_load(custom_ckpt))
                self.wte.weight.copy_(_get_any(sd, ("transformer.wte.weight", "wte.weight")))
                self.wpe.weight.copy_(_get_any(sd, ("transformer.wpe.weight", "wpe.weight")))
                self.ln_f.weight.copy_(_get_any(sd, ("transformer.ln_f.weight", "ln_f.weight")))
                self.ln_f.bias.copy_(_get_any(sd, ("transformer.ln_f.bias",   "ln_f.bias")))
                for i, blk in enumerate(self.blocks):
                    blk.ln1.weight.copy_(sd[f"transformer.h.{i}.ln_1.weight"]); blk.ln1.bias.copy_(sd[f"transformer.h.{i}.ln_1.bias"])
                    blk.ln3.weight.copy_(sd[f"transformer.h.{i}.ln_2.weight"]); blk.ln3.bias.copy_(sd[f"transformer.h.{i}.ln_2.bias"])
                    blk.sa.c_attn.weight.copy_(sd[f"transformer.h.{i}.attn.c_attn.weight"]); blk.sa.c_attn.bias.copy_(sd[f"transformer.h.{i}.attn.c_attn.bias"])
                    blk.sa.c_proj.weight.copy_(sd[f"transformer.h.{i}.attn.c_proj.weight"]);   blk.sa.c_proj.bias.copy_(sd[f"transformer.h.{i}.attn.c_proj.bias"])
                    blk.mlp.fc.weight.copy_(sd[f"transformer.h.{i}.mlp.c_fc.weight"]);         blk.mlp.fc.bias.copy_(sd[f"transformer.h.{i}.mlp.c_fc.bias"])
                    blk.mlp.proj.weight.copy_(sd[f"transformer.h.{i}.mlp.c_proj.weight"]);     blk.mlp.proj.bias.copy_(sd[f"transformer.h.{i}.mlp.c_proj.bias"])

        if freeze_lm:
            for p in self.parameters(): p.requires_grad_(False)
            for p in self.img_proj.parameters(): p.requires_grad_(True)
            for blk in self.blocks:
                for p in blk.ca.parameters():  p.requires_grad_(True)
                for p in blk.ln2.parameters(): p.requires_grad_(True)

    def forward(self, patch_tokens, input_ids, labels=None):
        """
        patch_tokens: (B, N_img, D_enc)
        input_ids:    (B, T_txt)
        """
        B, T = input_ids.shape
        device = input_ids.device

        img_mem = self.img_proj(patch_tokens)               
        pos = torch.arange(T, device=device).long()
        x = self.wte(input_ids) + self.wpe(pos)[None, :, :]   

        for blk in self.blocks:
            x = blk(x, img_mem)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            V = logits.size(-1)
            logits_shift = logits[:, :-1, :].contiguous()
            labels_shift = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits_shift.view(-1, V),
                labels_shift.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, patch_tokens, prompt_ids, max_new_tokens=32, temperature=0.8, top_p=0.95):
        seq = prompt_ids.clone()
        for _ in range(max_new_tokens):
            ctx = seq[:, -self.block_size:] if seq.size(1) > self.block_size else seq
            logits, _ = self.forward(patch_tokens, ctx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = next_logits.softmax(-1)

            if top_p < 1.0:
                sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_p.cumsum(dim=-1)
                cut = (cumsum > top_p).float().argmax(dim=-1).clamp_min(1)
                idx_next = []
                for b in range(probs.size(0)):
                    k = int(cut[b].item())
                    keep = sorted_idx[b, :k]
                    p = probs[b, keep]; p = p / p.sum()
                    idx_next.append(keep[torch.multinomial(p, 1)])
                next_tok = torch.stack(idx_next, dim=0)
            else:
                next_tok = torch.multinomial(probs, num_samples=1)

            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.tokenizer.eos_token_id).all():
                break
        return self.tokenizer.batch_decode(seq, skip_special_tokens=True)
