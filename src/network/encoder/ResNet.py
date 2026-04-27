from torch import nn
import lightning as L

import torch
import torch.nn as nn
import lightning as L

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels)
        )

    def forward(self, x):
        return torch.nn.functional.silu(x + self.net(x))
    

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        
        self.input_res = input_res
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        flatten_dim = d3 * self.last_h * self.last_w

        self.encoder_net = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(3, d1, 4, stride=2, padding=1),
            ResidualBlock(d1),
            nn.SiLU(),
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(d1, d2, 4, stride=2, padding=1),
            ResidualBlock(d2),
            nn.SiLU(),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(d2, d3, 4, stride=2, padding=1),
            ResidualBlock(d3),
            nn.SiLU(),
            
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def encode(self, x):
        return self.encoder_net(x)

    def forward(self, x):
        return self.encode(x)

class Decoder(L.LightningModule):
    def __init__(self, world_model_dim, model_dim, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        
        self.input_res = input_res
        self.model_dim = model_dim
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        flatten_dim = d3 * self.last_h * self.last_w
        
        self.decoder_input = nn.Sequential(
            nn.Linear(world_model_dim, flatten_dim),
            nn.SiLU()
        )

        self.decoder_net = nn.Sequential(
            # Up-sample 1: 8x8 -> 16x16
            ResidualBlock(d3),
            nn.ConvTranspose2d(d3, d2, 4, stride=2, padding=1),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
            # Up-sample 2: 16x16 -> 32x32
            ResidualBlock(d2),
            nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=1),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
            # Up-sample 3: 32x32 -> 64x64
            ResidualBlock(d1),
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.model_dim, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        return self.decode(x)