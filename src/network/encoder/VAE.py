import torch
from torch import nn
import lightning as L

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int, input_res=(64, 64), final_res: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.input_res = input_res
        # Scalatura canali
        d1, d2, d3, d4 = model_dim // 4, model_dim // 2, model_dim, model_dim
        
        self.final_res = final_res

        if final_res == 4:
            # 4 strati: 64 -> 32 -> 16 -> 8 -> 4
            self.encoder_net = nn.Sequential(
                nn.Conv2d(3, d1, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(d1, d2, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(d2, d3, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(d3, d4, 4, stride=2, padding=1), nn.SiLU(),
            )
            self.out_channels = d4
        else:
            # Configurazione per arrivare a 6x6 (3 strati)
            self.encoder_net = nn.Sequential(
                nn.Conv2d(3, d1, 4, stride=2, padding=0), nn.SiLU(),
                nn.Conv2d(d1, d2, 4, stride=2, padding=0), nn.SiLU(),
                nn.Conv2d(d2, d3, 4, stride=2, padding=0), nn.SiLU(),
            )
            self.out_channels = d3

        flatten_dim = self.out_channels * final_res * final_res
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    def encode(self, x):
        """Metodo esplicito per la codifica"""
        return self.projection(self.encoder_net(x))
    
    def forward(self, x):
        return self.projection(self.encoder_net(x))

class Decoder(L.LightningModule):
    def __init__(self, world_model_dim: int, model_dim: int, input_res=(64, 64), final_res: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.input_res = input_res
        d1, d2, d3, d4 = model_dim // 4, model_dim // 2, model_dim, model_dim
        self.final_res = final_res
        self.out_channels = d4 if final_res == 4 else d3
        
        flatten_dim = self.out_channels * final_res * final_res
        self.decoder_input = nn.Linear(world_model_dim, flatten_dim)

        if final_res == 4:
            self.decoder_net = nn.Sequential(
                nn.ConvTranspose2d(d4, d3, 4, stride=2, padding=1), nn.SiLU(),
                nn.ConvTranspose2d(d3, d2, 4, stride=2, padding=1), nn.SiLU(),
                nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=1), nn.SiLU(),
                nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        else:
            self.decoder_net = nn.Sequential(
                nn.ConvTranspose2d(d3, d2, 4, stride=2, padding=0), nn.SiLU(),
                nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=0), nn.SiLU(),
                nn.ConvTranspose2d(d1, 3, 6, stride=2, padding=0),
                nn.Sigmoid()
            )
    def decode(self, z):
        """Metodo esplicito per la decodifica"""
        x = self.decoder_input(z)
        x = x.view(-1, self.out_channels, self.final_res, self.final_res)
        return self.decoder_net(x)
    
    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.out_channels, self.final_res, self.final_res)
        return self.decoder_net(x)