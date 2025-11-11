import os, torch
from PIL import Image, ImageDraw, ImageFont
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import GPT2TokenizerFast
from model import CaptionGPT2

device = "cuda" if torch.cuda.is_available() else "cpu"

BEST = "/Data/gabriel.lenain/image-captioner/runs/xattn_gpt2_dino_custom/cap_xattn_best.pt"
CUSTOM_GPT = "/users/eleves-b/2022/gabriel.lenain/ZerotoHero/log/model_19073_flat.pt"
IMG = "/Data/gabriel.lenain/image-captioner/datasets/coco2017_full/val_images/000000000285.jpg"
OUT = "/Data/gabriel.lenain/image-captioner/runs/xattn_gpt2_dino_custom/000000000285_captioned.png"

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def overlay_caption(image_path, caption, out_path, max_w=1280, pad=12):
    im = Image.open(image_path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()

    # wrap caption
    draw = ImageDraw.Draw(im)
    max_chars = max(20, im.width // 14)
    words, lines = caption.split(), []
    for w in words:
        if not lines: lines = [w]
        elif len(lines[-1]) + 1 + len(w) <= max_chars: lines[-1] += " " + w
        else: lines.append(w)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + 6
    strip_h = pad + len(lines) * line_h + pad

    canvas = Image.new("RGB", (im.width, im.height + strip_h), (0, 0, 0))
    canvas.paste(im, (0, 0))
    y = im.height + pad
    for line in lines:
        w = draw.textlength(line, font=font)
        ImageDraw.Draw(canvas).text(((im.width - w) // 2, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"[saved] {out_path}")

@torch.no_grad()
def main():
    assert os.path.exists(BEST), BEST
    assert os.path.exists(CUSTOM_GPT), CUSTOM_GPT
    assert os.path.exists(IMG), IMG

    ckpt = safe_load(BEST)
    state = ckpt.get("model_state", ckpt)
    enc_dim = state["img_proj.weight"].shape[1] 
    gpt_name = ckpt.get("gpt_name", "gpt2")

    if enc_dim == 1024:
        vit_name = "vit_large_patch14_dinov2.lvd142m"
    elif enc_dim == 768:
        vit_name = "vit_base_patch14_dinov2.lvd142m"
    else:
        raise RuntimeError(f"Unsupported enc_dim {enc_dim}; choose a vision encoder that outputs this dim.")

    vit = timm.create_model(vit_name, pretrained=True).to(device).eval()
    tfm = create_transform(**resolve_data_config({}, model=vit))

    img = tfm(Image.open(IMG).convert("RGB")).unsqueeze(0).to(device)  
    feats = vit.forward_features(img)
    if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
        patch_tokens = feats["x_norm_patchtokens"] 
    else:
        x = feats if isinstance(feats, torch.Tensor) else feats.get("tokens")
        assert x.dim() == 3 and x.size(1) > 1, "Cannot extract patch tokens."
        patch_tokens = x[:, 1:, :] 

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = CaptionGPT2(
        enc_dim=enc_dim,
        tokenizer=tok,
        gpt_name=gpt_name,
        freeze_lm=False,
        custom_ckpt=CUSTOM_GPT,
    ).to(device).eval()
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded. missing={len(missing)} unexpected={len(unexpected)}")

    prompt_ids = torch.tensor([tok.encode("A photo of")], device=device)
    caption = model.generate(patch_tokens, prompt_ids, max_new_tokens=32, temperature=0.8, top_p=0.95)[0]
    print(caption)

    overlay_caption(IMG, caption, OUT)

if __name__ == "__main__":
    main()
