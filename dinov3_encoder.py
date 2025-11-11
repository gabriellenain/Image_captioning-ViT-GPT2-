import torch
import torch.nn as nn

class PatchEncoder(nn.Module):
    """
    Returns ViT patch tokens (no CLS): (B, N, enc_dim)
    Exposes `preprocess` for consistent resize/normalize.
    Prefers external dinov3_loader if available, else timm DINOv2.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.using_external = False
        self.external_fn = None
        self.preprocess = None
        self.enc_dim = None

        try:
            from dinov3_loader import dinov3_patch_tokens, DINOv3_DIM, DINOV3_PREPROCESS
            self.external_fn = dinov3_patch_tokens
            self.enc_dim = int(DINOv3_DIM)
            self.preprocess = DINOV3_PREPROCESS  
            self.using_external = True
            print("[PatchEncoder] Using external dinov3_loader.")
        except Exception:
            import timm
            print("[PatchEncoder] Using timm: vit_large_patch14_dinov2.lvd142m")
            self.backbone = timm.create_model(
                "vit_large_patch14_dinov2.lvd142m",
                pretrained=True, num_classes=0
            ).eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.enc_dim = int(self.backbone.num_features)
            data_cfg = timm.data.resolve_model_data_config(self.backbone)
            self.preprocess = timm.data.create_transform(**data_cfg, is_training=False)

        self.to(self.device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,H,W) already preprocessed & on self.device.
        returns: (B, N, enc_dim) patch tokens (CLS removed)
        """
        images = images.to(self.device, non_blocking=True)
        if self.using_external:
            return self.external_fn(images)  

        feats = self.backbone.forward_features(images)  
        x = feats["x"] if isinstance(feats, dict) and "x" in feats else feats
        return x[:, 1:, :]  
