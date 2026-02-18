import os
import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms.v2 import functional as F
import torchvision
import lightning as L

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int = 512, unimix: float = 0.01):
        """
        VAE ottimizzato per VizDoom 160x120.
        Mantiene la struttura standalone ma potenzia la capacità visiva per piccoli oggetti.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Configurazione canali DreamerV3
        # model_dim=512 -> base_depth=32
        base_depth = model_dim // 16
        d1 = base_depth * 1  # 32
        d2 = base_depth * 2  # 64
        d3 = base_depth * 4  # 128
        d4 = base_depth * 8  # 256
        
        # --- ENCODER ---
        # Input: 3 x 120 x 160
        self.encoder = nn.Sequential(
            # Layer 1: 3 -> d1 | Stride 2 | Out: 60 x 80
            nn.Conv2d(3, d1, 4, 2, 1, bias=False),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
            # Layer 2: d1 -> d2 | Stride 2 | Out: 30 x 40
            nn.Conv2d(d1, d2, 4, 2, 1, bias=False),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
            # Layer 3: d2 -> d3 | Stride 2 | Out: 15 x 20
            # Qui raggiungiamo il limite per l'altezza (15 è dispari)
            nn.Conv2d(d2, d3, 4, 2, 1, bias=False),
            nn.GroupNorm(1, d3),
            nn.SiLU(),
            
            # Layer 4: d3 -> d4 | STRIDE 1 (Cruciale!) | Out: 15 x 20
            # Non riduciamo ulteriormente per non perdere i nemici piccoli
            nn.Conv2d(d3, d4, 3, 1, 1, bias=False),
            nn.GroupNorm(1, d4),
            nn.SiLU()
        )

        # Dimensioni esatte del bottleneck visivo
        self.last_conv_height = 15
        self.last_conv_width = 20
        self.num_filters = d4 
        
        # Flatten: 256 * 15 * 20 = 76,800 features
        # Manteniamo alta la risoluzione spaziale invece di comprimere
        flatten_dim = self.num_filters * self.last_conv_height * self.last_conv_width
        
        # --- BOTTLENECK ---
        mlp_hidden = model_dim * 2 
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(flatten_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden), # LayerNorm standard su vettori 1D
            nn.SiLU()
        )

        self.mean_predictor = nn.Linear(mlp_hidden, latent_dim)
        self.variance_predictor = nn.Linear(mlp_hidden, latent_dim)

        # --- DECODER EXPANSION ---
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, flatten_dim),
            nn.SiLU()
        )

        # --- DECODER ---
        # Speculare all'Encoder per tornare esattamente a 120x160
        self.decoder = nn.Sequential(
            # Layer 1: d4 -> d3 | Stride 1 | In: 15x20 -> Out: 15x20
            nn.ConvTranspose2d(d4, d3, 3, 1, 1, bias=False),
            nn.GroupNorm(1, d3),
            nn.SiLU(),
            
            # Layer 2: d3 -> d2 | Stride 2 | In: 15x20 -> Out: 30x40
            nn.ConvTranspose2d(d3, d2, 4, 2, 1, bias=False),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
            # Layer 3: d2 -> d1 | Stride 2 | In: 30x40 -> Out: 60x80
            nn.ConvTranspose2d(d2, d1, 4, 2, 1, bias=False),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
            # Layer 4: d1 -> 3 | Stride 2 | In: 60x80 -> Out: 120x160
            nn.ConvTranspose2d(d1, 3, 4, 2, 1)
        )

        # Inizializzazione pesi (opzionale ma consigliata)
        nn.init.xavier_uniform_(self.mean_predictor.weight)
        nn.init.constant_(self.mean_predictor.bias, 0)

    def encode(self, x):
        # add ajustaments ?...
        x_encoded = self.encoder(x)
        x_flat = x_encoded.reshape(x_encoded.size(0), -1)
        hidden = self.fc_encoder(x_flat)
        mean = self.mean_predictor(hidden)
        logvar = self.variance_predictor(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z):
        z_proj = self.fc_decoder(z)
        z_proj = z_proj.view(-1, self.num_filters, self.last_conv_height, self.last_conv_width)
        x_hat = self.decoder(z_proj)
        # Non serve più interpolate o crop: l'output è matematicamente 120x160
        return x_hat

    def forward(self, x):
        # Gestione input uint8 o float
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        
        return x_hat, mean, logvar

    def _compute_loss(self, x, x_hat, mean, logvar):
        # Nota: La loss la calcoli esternamente come richiesto, 
        # ma qui mantengo la firma per compatibilità col tuo training step
        recon_loss = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        batch_size = x.size(0)
        return recon_loss / batch_size, kl_loss / batch_size

    def training_step(self, batch, batch_idx):
        # Normalizza qui se il dataloader passa uint8
        x = batch.float() / 255.0 if batch.max() > 1.0 else batch
        
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z) 
        recon_loss, kl_loss = self._compute_loss(x, x_hat, mean, logvar)
        
        # Per vedere i nemici piccoli, spesso aiuta dare un peso minore alla KL all'inizio
        loss = recon_loss + 0.001 * kl_loss
        
        self.log_dict({"train_loss": loss, "train_recon": recon_loss, "train_kl": kl_loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.float() / 255.0 if batch.max() > 1.0 else batch
        mean, logvar = self.encode(x)
        z = mean
        x_hat_logits = self.decode(z)
        recon_loss, kl_loss = self._compute_loss(x, x_hat_logits, mean, logvar)
        val_loss = recon_loss + 0.001 * kl_loss
        self.log("val_loss", val_loss, prog_bar=True)
        
        if batch_idx == 0:
            x_hat_img = torch.sigmoid(x_hat_logits)
            n = min(batch.size(0), 8)
            grid = torchvision.utils.make_grid(torch.cat([x[:n], x_hat_img[:n]]), nrow=n)
            self.logger.experiment.add_image('val_reconstruction', grid, self.current_epoch)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)