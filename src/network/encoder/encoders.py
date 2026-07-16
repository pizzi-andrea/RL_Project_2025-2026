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


class ImpalaBlock(nn.Module):
    """
    Un blocco IMPALA standard è composto da:
    1. Una convoluzione per proiettare i canali alla dimensione desiderata.
    2. Un MaxPool per dimezzare la risoluzione spaziale.
    3. Due blocchi residui consecutivi.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # MaxPool2d dimezza la risoluzione spaziale ad ogni blocco (es. da 64 a 32)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        # Manteniamo la scalabilità dinamica dei canali basata su model_dim
        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        
        self.input_res = input_res
        # Con 3 blocchi IMPALA (ognuno con un maxpool a dimezzare), la risoluzione viene divisa per 8
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        flatten_dim = d3 * self.last_h * self.last_w

        self.encoder_net = nn.Sequential(
            # Layer 1: 3 -> d1 | Dimezza risoluzione (es. 64x64 -> 32x32 o 160x120 -> 80x60)
            ImpalaBlock(3, d1),
            
            # Layer 2: d1 -> d2 | Dimezza risoluzione (es. 32x32 -> 16x16 o 80x60 -> 40x30)
            ImpalaBlock(d1, d2),
            
            # Layer 3: d2 -> d3 | Dimezza risoluzione (es. 16x16 -> 8x8 o 40x30 -> 20x15)
            ImpalaBlock(d2, d3),
            
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def encode(self, x):
        return self.encoder_net(x)

    def forward(self, x):
        return self.encode(x)
    
    

class ResNet(L.LightningModule):
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
    


class Encoder(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        # Limitiamo i canali convoluzionali interni a un massimo di 128.
        # Anche con model_dim=512, la CNN rimarrà leggerissima e non saturerà la VRAM,
        # mentre l'output finale dello strato Linear sarà correttamente di dimensione 'latent_dim'.
        self.cnn_dim = min(model_dim, 128)
        d1 = max(8, self.cnn_dim // 4)
        d2 = max(16, self.cnn_dim // 2)
        d3 = self.cnn_dim
        
        self.input_res = input_res
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        flatten_dim = d3 * self.last_h * self.last_w

        self.encoder_net = nn.Sequential(
            # Layer 1: Downsample 1/2 (es. 160x120 -> 80x60 o 64x64 -> 32x32)
            nn.Conv2d(3, d1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
            # Layer 2: Downsample 1/4 (es. 80x60 -> 40x30 o 32x32 -> 16x16)
            nn.Conv2d(d1, d2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
            # Layer 3: Downsample 1/8 (es. 40x30 -> 20x15 o 16x16 -> 8x8)
            nn.Conv2d(d2, d3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d3),
            nn.SiLU(),
            
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def encode(self, x):
        return self.encoder_net(x)

    def forward(self, x):
        return self.encode(x)