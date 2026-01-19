import os
from torch.optim import Adam
from pathlib import Path
from torch import nn
from torchvision.utils import save_image
import lightning as L
import torch
import torchvision
from random import randint
torch.set_float32_matmul_precision('high')

class VAE(L.LightningModule):
    def __init__(self, latent_dim:int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 320x240 -> 160x120
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 160x120 -> 80x60
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 80x60 -> 40x30
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 40x30 -> 20x15
            nn.ReLU()
        )

        # ---------------- Latent layers ----------------
        self.flatten_dim = 256 * 15 * 20
        self.mean_predictor = nn.Linear(self.flatten_dim, latent_dim)
        self.variance_predictor = nn.Linear(self.flatten_dim, latent_dim)
        self.z_project = nn.Linear(latent_dim, self.flatten_dim)         # per il decoder

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 20x15 -> 40x30
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 40x30 -> 80x60
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 80x60 -> 160x120
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 160x120 -> 320x240
            nn.Sigmoid()
        )
        
        self.save_hyperparameters()
        
        
    
    def setup(self, stage:str):
        pass
    
    # copy by https://github.com/kuc2477/pytorch-vae/blob/master/model.py
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(reduction="sum")(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self,x, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)

    def training_step(self, batch, batch_idx):
        y, mean, var = self.forward(batch)

        recon_loss = self.reconstruction_loss(y, batch)
        kl_loss = self.kl_divergence_loss(y, mean, var)

        loss = recon_loss + 0.001 * kl_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_recon": recon_loss,
                "train_kl": kl_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        start = randint(0, len(batch) - 3)
        y , mean, var = self.forward(batch)
        sample_imgs = batch[start:start+3]
        y_sample = y[start:start+3]
        
        grid_true = torchvision.utils.make_grid(nrow=3, tensor=sample_imgs)
        grid_pred = torchvision.utils.make_grid(nrow=3, tensor=y_sample)

        grid = torch.hstack((grid_true, grid_pred))
        log_dir = self.trainer.logger.log_dir or self.trainer.default_root_dir
    
        # Costruisci il nome file
        filename = f"grid_epoch_{self.current_epoch}.png"
        save_path = os.path.join(log_dir, filename)
        
        save_image(grid, save_path)
        #self.logger.experiment.add_image('example_images', grid, 0)

        recon_loss = self.reconstruction_loss(y, batch)
        kl_loss = self.kl_divergence_loss(y, mean, var)

        loss = recon_loss + 0.001 * kl_loss

        self.log_dict(
            {
                "validation_loss": loss,
                "validation_recon": recon_loss,
                "validation_kl": kl_loss,
            },
            prog_bar=True,
        )
    
    def test_step(self, batch, batch_idx):
        pass
    
    def predict_step(self, batch, batch_idx):
        y, mean, var = self.forward(batch)
        
        #self.logger.experiment.add_image('example_images', grid, 0)
        return (y, mean)

        

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    
    def forward(self, x):
        encode_img = self.encoder(x)
        mean_var = encode_img.view(encode_img.size(0), -1)

        mean = self.mean_predictor(mean_var)
        var = self.variance_predictor(mean_var)

        # reparameterization trick
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        if self.training:
            z = mean + std * eps
        else:
            z = mean

        
        z_j = self.z_project(z)
        z_j = z_j.view(-1, 256, 15, 20)
        return self.decoder(z_j), mean, var  # no need for extra sigmoid, already in decoder
        




    
    
        

    







        
