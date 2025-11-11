import torch
DINOv3_DIM = 1024
@torch.no_grad()
def dinov3_patch_tokens(images: torch.Tensor) -> torch.Tensor:
    B = images.size(0); N = 196  
    return torch.zeros(B, N, DINOv3_DIM, device=images.device)
