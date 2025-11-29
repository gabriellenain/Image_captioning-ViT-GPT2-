import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)      
        tok_emb = self.transformer.wte(idx)     
        x = tok_emb + pos_emb                   
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

def load_nanogpt_from_ckpt(ckpt_path: str, map_location="cpu") -> GPT:
    raw = torch.load(ckpt_path, map_location=map_location)
    if isinstance(raw, dict) and "model" in raw and "config" in raw:
        sd = raw["model"]
        cfg = raw["config"]
        print(f"[NanoGPT] Loaded checkpoint {ckpt_path} "
              f"(step={raw.get('step','?')}, val_loss={raw.get('val_loss','?')})")
    else:
        sd = raw
        cfg = GPTConfig(vocab_size=50304)  # ton cas
        print(f"[NanoGPT] Loaded flat state_dict from {ckpt_path}")

    gpt = GPT(cfg)
    gpt.load_state_dict(sd, strict=True)
    return gpt


class CaptionNanoGPT(nn.Module):
    """
    LLaVA-style bridge:
      - simple learned linear projection enc_dim -> n_embd
      - projected image tokens are used directly as prefix tokens
        before the text tokens in the GPT.

    - patch_tokens : (B, N_img, enc_dim) venant de CLIP (gelé)
    - NanoGPT (gelé) prend [image_prefix, texte] via embeddings + wpe
    """

    def __init__(
        self,
        enc_dim: int,
        tokenizer,          
        custom_ckpt: str,
        m_vis_tokens: int = 32,  # not used anymore; image token count comes from encoder
        use_cls_only: bool = False,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_cls_only = use_cls_only

        self.gpt = load_nanogpt_from_ckpt(custom_ckpt)
        cfg = self.gpt.config
        self.d = cfg.n_embd
        self.block_size = cfg.block_size

        self.img_proj = nn.Linear(enc_dim, self.d)

        self.wte = self.gpt.transformer.wte
        self.wpe = self.gpt.transformer.wpe

        if freeze_lm:
            for p in self.gpt.parameters():
                p.requires_grad_(False)

    def _decode_transformer(self, full_embeds):
        """Passe les embeddings (B, L, d) dans les blocs GPT."""
        x = full_embeds
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        logits = self.gpt.lm_head(x)
        return logits

    def forward(self, patch_tokens, input_ids, labels=None):
        """
        patch_tokens : (B, N_img, enc_dim)
        input_ids    : (B, T_txt)
        labels       : (B, T_txt) (-100 sur pads)
        """
        B, T_txt = input_ids.shape
        device = input_ids.device

        if patch_tokens.dim() == 2:
            patch_tokens = patch_tokens.unsqueeze(1)
        B_img, N_raw, D_enc = patch_tokens.shape
        assert B_img == B, "batch size image != batch size texte"

        x_img = patch_tokens
        if self.use_cls_only:
            x_img = x_img[:, 0:1, :]  # (B, 1, enc_dim)

        img_embeds = self.img_proj(x_img)  # (B, N_img_or_1, d)
        M = img_embeds.size(1)

        txt_embeds = self.wte(input_ids)  # (B, T_txt, d)

        full_len = M + T_txt
        if full_len > self.block_size:
            cut_txt = self.block_size - M
            txt_embeds = txt_embeds[:, :cut_txt, :]
            input_ids = input_ids[:, :cut_txt]
            if labels is not None:
                labels = labels[:, :cut_txt]
            T_txt = cut_txt
            full_len = M + T_txt

        pos = torch.arange(full_len, device=device)
        pos_embeds = self.wpe(pos).unsqueeze(0)     # (1, full_len, d)

        full_embeds = torch.cat([img_embeds, txt_embeds], dim=1)  # (B, full_len, d)
        full_embeds = full_embeds + pos_embeds

        logits = self._decode_transformer(full_embeds)

        loss = None
        if labels is not None:
            # On ne calcule la perte que sur la partie texte
            logits_text = logits[:, M:M+T_txt, :]

            shift_logits = logits_text[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        patch_tokens,
        prompt_ids,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """
        Génération auto-régressive avec NanoGPT + prefix image.
        """
        self.gpt.eval()
        device = prompt_ids.device
        B = patch_tokens.size(0)

        if patch_tokens.dim() == 2:
            patch_tokens = patch_tokens.unsqueeze(1)
        B_img, N_raw, D_enc = patch_tokens.shape
        assert B_img == B

        x_img = patch_tokens
        if self.use_cls_only:
            x_img = x_img[:, 0:1, :]

        img_embeds = self.img_proj(x_img)  # (B, N_img_or_1, d)
        M = img_embeds.size(1)

        seq = prompt_ids.clone()

        for _ in range(max_new_tokens):
            T_txt = seq.size(1)
            full_len = M + T_txt
            if full_len > self.block_size:
                cut_txt = self.block_size - M
                seq_ctx = seq[:, -cut_txt:]
                T_txt = cut_txt
            else:
                seq_ctx = seq

            txt_embeds = self.wte(seq_ctx)  # (B, T_txt, d)
            L = M + T_txt
            pos = torch.arange(L, device=device)
            pos_embeds = self.wpe(pos).unsqueeze(0)  # (1, L, d)

            full_embeds = torch.cat([img_embeds, txt_embeds], dim=1)
            full_embeds = full_embeds + pos_embeds

            logits = self._decode_transformer(full_embeds)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = next_logits.softmax(dim=-1)

            if top_p < 1.0:
                sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_p.cumsum(dim=-1)
                cutoff = (cumsum > top_p).float().argmax(dim=-1).clamp_min(1)
                next_tok = []
                for b in range(B):
                    k = int(cutoff[b].item())
                    keep = sorted_idx[b, :k]
                    p = probs[b, keep]
                    p = p / p.sum()
                    next_tok.append(keep[torch.multinomial(p, 1)])
                next_tok = torch.stack(next_tok, dim=0)
            else:
                next_tok = torch.multinomial(probs, num_samples=1)

            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok.squeeze(1) == self.tokenizer.eos_token_id).all():
                break

        return self.tokenizer.batch_decode(seq, skip_special_tokens=True)
