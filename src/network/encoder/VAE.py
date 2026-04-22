from torch import nn
import lightning as L


class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int = 128, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        d4 = model_dim * 2

        self.last_h = input_res[0] // 16
        self.last_w = input_res[1] // 16
        flatten_dim = d4 * self.last_h * self.last_w

        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, d1, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(d1, d2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(d2, d3, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(d3, d4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
        )

        self.decoder_input = nn.Linear(latent_dim, flatten_dim)

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(d4, d3, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(d3, d2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
        )

    def encode(self, x):
        # x: [B, 3, H, W] → (embed: [B, latent_dim], None)
        # Ritorna tupla per compatibilità con VAEEncoderAdapter (mean, _)
        return self.encoder_net(x), None

    def decode(self, z):
        # z: [B, latent_dim] → [B, 3, H, W] logits
        result = self.decoder_input(z)
        result = result.view(-1, self.hparams.model_dim * 2, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        embed, _ = self.encode(x)
        return embed