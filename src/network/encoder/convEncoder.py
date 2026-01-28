# -------------------------
# CNN Encoder/Decoder for 64x64
# -------------------------
from torch import nn

import torch
class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.SiLU(),   # 64 -> 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),     # 32 -> 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),    # 16 -> 8
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),   # 8 -> 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] or [B,C,H,W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape #T is the lenght of the temporal sequence
            x = x.reshape(B * T, C, H, W)
            y = self.net(x)
            return y.view(B, T, -1) # [B*T, 1024] → [B, T, 1024]

        return self.net(x)

class ConvDecoder(nn.Module):
    def __init__(self, in_dim: int, out_ch: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256 * 4 * 4),
            nn.SiLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),    # 16->32
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),           # 32->64
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,T,D] or [B,D]
        if feat.dim() == 3:
            B, T, D = feat.shape
            z = self.fc(feat.view(B * T, D)).view(B * T, 256, 4, 4)
            x = self.deconv(z).view(B, T, -1, 64, 64)
            return x
        z = self.fc(feat).view(-1, 256, 4, 4)
        return self.deconv(z) #here deterministic decoder, not probabilistic
