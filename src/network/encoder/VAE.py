import os
import torch
from torch import nn
from torch.optim import Adam
import torchvision
import lightning as L

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.save_hyperparameters()
        
        """Simple convolutional encoder for fixed size image"""
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        """Emb computation"""
        self.flatten_dim = 256 * 15 * 20
        self.mean_predictor = nn.Linear(self.flatten_dim, latent_dim)
        self.variance_predictor = nn.Linear(self.flatten_dim, latent_dim)
        self.z_project = nn.Linear(latent_dim, self.flatten_dim)

        """Simple decoder convolutional"""
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            
        )

    def encode(self, x):
        """Encode image in latent distribution"""
        x_encoded = self.encoder(x)
        x_flat = x_encoded.view(x_encoded.size(0), -1)
        mean = self.mean_predictor(x_flat)
        logvar = self.variance_predictor(x_flat)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Make sample reparametrizable"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z):
        """Decode latent vector in image space"""
        z_proj = self.z_project(z)
        z_proj = z_proj.view(-1, 256, 15, 20)
        x_hat = self.decoder(z_proj)
        return x_hat

    def forward(self, x):
        """Forward for inference"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def _compute_loss(self, x, x_hat, mean, logvar):
        
        recon_loss = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
        
        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        
        batch_size = x.size(0)
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size
        
        return recon_loss, kl_loss

    def training_step(self, batch, batch_idx):

        # Eseguiamo i passaggi esplicitamente qui invece che in forward per chiarezza nel training
        mean, logvar = self.encode(batch)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z) 

        recon_loss, kl_loss = self._compute_loss(batch, x_hat, mean, logvar)
        loss = recon_loss + 0.001 * kl_loss

        self.log_dict({
            "train_loss": loss,
            "train_recon": recon_loss,
            "train_kl": kl_loss
        }, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        

        
    
        mean, logvar = self.encode(batch)
        z = mean # In validazione usiamo la media, non il campionamento casuale solitamente
        x_hat_logits = self.decode(z)
        
        recon_loss, kl_loss = self._compute_loss(batch, x_hat_logits, mean, logvar)
        val_loss = recon_loss + 0.001 * kl_loss

        self.log("val_loss", val_loss, prog_bar=True)

        # Salvataggio immagini: SOLO sul primo batch per evitare I/O inutile
        if batch_idx == 0:
            x_hat_img = torch.sigmoid(x_hat_logits) # Applichiamo sigmoid per visualizzazione
            n = min(batch.size(0), 8)
            grid = torchvision.utils.make_grid(torch.cat([batch[:n], x_hat_img[:n]]), nrow=n)
            
            # Log su tensorboard/logger
            self.logger.experiment.add_image('val_reconstruction', grid, self.current_epoch)
            
            # local log
            log_dir = self.trainer.log_dir or "."
            save_path = os.path.join(log_dir, f"epoch_{self.current_epoch}.png")
            torchvision.utils.save_image(grid, save_path)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)