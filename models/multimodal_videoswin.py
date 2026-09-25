"""
Encoder visual de la rama Swin3D (SF-MM, SIMBig 2026).

Lo importan train_swin3d_standalone_fits.py y los scripts que cargan sus
checkpoints. Se conserva el atributo `swin` para que los nombres de los pesos
coincidan con los checkpoints entrenados (model.video_enc.swin.*).
"""
import torch
import torch.nn as nn


class VideoSwinEncoder(nn.Module):
    """
    VideoSwin-T (Kinetics-400 pretrained) como backbone visual.

    Input : [B, 3, T, H, W]   — T=16, H=W=224
    Output: [B, video_out_dim]
    """

    def __init__(self, video_out_dim: int = 768, pretrained: bool = True):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights

        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        self.swin = swin3d_t(weights=weights)
        self.swin.head = nn.Linear(768, video_out_dim)
        nn.init.trunc_normal_(self.swin.head.weight, std=0.02)
        nn.init.zeros_(self.swin.head.bias)
        self.video_out_dim = video_out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swin(x)

