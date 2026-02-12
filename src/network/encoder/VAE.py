import os
import torch
from torch import nn
from torch.optim import Adam
import torchvision
import lightning as L

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int = 512):
        """
        VAE compliant with DreamerV3 Model Sizes.
        
        Args:
            latent_dim (int): Dimension of the stochastic latent state (z).
            model_dim (int): The model dimension 'd' from the paper.
                             Base CNN channels will be d / 16.
                             Default 512 corresponds to ~50M param model range.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # According to DreamerV3, Base CNN channels = d / 16
        # Multipliers are usually [1, 2, 4, 8] for the 4 layers
        base_depth = model_dim // 16
        d1 = base_depth * 1
        d2 = base_depth * 2
        d3 = base_depth * 4
        d4 = base_depth * 8
        
        # --- ENCODER ---
        self.encoder = nn.Sequential(
            # Layer 1: 3 -> d1
            nn.Conv2d(3, d1, 4, 2, 1),
            nn.BatchNorm2d(d1),
            nn.SiLU(),
            # Layer 2: d1 -> d2
            nn.Conv2d(d1, d2, 4, 2, 1),
            nn.BatchNorm2d(d2),
            nn.SiLU(),
            # Layer 3: d2 -> d3
            nn.Conv2d(d2, d3, 4, 2, 1),
            nn.BatchNorm2d(d3),
            nn.SiLU(),
            # Layer 4: d3 -> d4
            nn.Conv2d(d3, d4, 4, 2, 1),
            nn.BatchNorm2d(d4),
            nn.SiLU()
        )

        # --- DECODER ---
        self.decoder = nn.Sequential(
            # Layer 1: d4 -> d3
            nn.ConvTranspose2d(d4, d3, 4, 2, 1, output_padding=(1, 0)), # Note: output_padding tuning might be needed based on exact resolution
            nn.BatchNorm2d(d3),
            nn.SiLU(),
            # Layer 2: d3 -> d2
            nn.ConvTranspose2d(d3, d2, 4, 2, 1),
            nn.BatchNorm2d(d2),
            nn.SiLU(),
            # Layer 3: d2 -> d1
            nn.ConvTranspose2d(d2, d1, 4, 2, 1),
            nn.BatchNorm2d(d1),
            nn.SiLU(),
            # Layer 4: d1 -> 3
            nn.ConvTranspose2d(d1, 3, 4, 2, 1)
        )
        
        # Assuming input is 120x160 based on your previous code:
        # 160 / 2^4 = 10
        # 120 / 2^4 = 7.5 -> 7 (via floor in Conv2d)
        self.last_conv_height = 7
        self.last_conv_width = 10
        self.num_filters = d4 # This is now dynamic: 8 * (model_dim // 16)
        
        flatten_dim = self.num_filters * self.last_conv_height * self.last_conv_width
        
        # --- BOTTLENECK ---
        # The MLP hidden units usually match model_dim in DreamerV3
        mlp_hidden = model_dim * 2 # Often slightly larger or equal to model_dim before split
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(flatten_dim, mlp_hidden),
            nn.RMSNorm(mlp_hidden),
            nn.SiLU()
        )

        self.mean_predictor = nn.Linear(mlp_hidden, latent_dim)
        self.variance_predictor = nn.Linear(mlp_hidden, latent_dim)

        # --- DECODER EXPANSION ---
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.RMSNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, flatten_dim),
            nn.SiLU()
        )

        # Initialization
        nn.init.xavier_uniform_(self.mean_predictor.weight)
        nn.init.constant_(self.mean_predictor.bias, 0)

    def encode(self, x):
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
        
        # Dynamic view based on num_filters
        z_proj = z_proj.view(-1, self.num_filters, self.last_conv_height, self.last_conv_width)
        x_hat = self.decoder(z_proj)
        
        # Enforce output resolution if dimensions mismatch due to padding/strides
        if x_hat.shape[-2:] != (120, 160):
            x_hat = nn.functional.interpolate(x_hat, size=(120, 160), mode='bilinear', align_corners=False)
        return x_hat

    def forward(self, x):
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        
        return x_hat, mean, logvar

    def _compute_loss(self, x, x_hat, mean, logvar):
        recon_loss = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        batch_size = x.size(0)
        return recon_loss / batch_size, kl_loss / batch_size

    def training_step(self, batch, batch_idx):
        mean, logvar = self.encode(batch)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z) 
        recon_loss, kl_loss = self._compute_loss(batch, x_hat, mean, logvar)
        
        # Added Beta schedule or fixed small KL weight is typical
        loss = recon_loss + 0.001 * kl_loss
        
        self.log_dict({"train_loss": loss, "train_recon": recon_loss, "train_kl": kl_loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mean, logvar = self.encode(batch)
        z = mean
        x_hat_logits = self.decode(z)
        recon_loss, kl_loss = self._compute_loss(batch, x_hat_logits, mean, logvar)
        val_loss = recon_loss + 0.001 * kl_loss
        self.log("val_loss", val_loss, prog_bar=True)
        if batch_idx == 0:
            x_hat_img = torch.sigmoid(x_hat_logits)
            n = min(batch.size(0), 8)
            grid = torchvision.utils.make_grid(torch.cat([batch[:n], x_hat_img[:n]]), nrow=n)
            self.logger.experiment.add_image('val_reconstruction', grid, self.current_epoch)
            log_dir = self.trainer.log_dir or "."
            # Optional save
            # save_path = os.path.join(log_dir, f"epoch_{self.current_epoch}.png")
            # torchvision.utils.save_image(grid, save_path)

    def configure_optimizers(self):
        # DreamerV3 standard LR is usually 1e-4 or 3e-4, 1e-3 is often too high for VAE
        return Adam(self.parameters(), lr=1e-3)