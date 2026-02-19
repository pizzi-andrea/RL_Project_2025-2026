import torch
from torch import nn
import lightning as L


class StandardResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Prima convoluzione classica 3x3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_channels) # 8 gruppi è un buon compromesso
        
        # Seconda convoluzione classica 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        self.relu = nn.ReLU(inplace=True) # ReLU standard per ResNet

        # Shortcut per match delle dimensioni
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.relu(x + residual)

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, model_dim: int = 512, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()
        
        # Canali potenziati per catturare meglio i dettagli del nemico
        # d1=64 è lo standard ResNet per iniziare a vedere bene i bordi
        d1, d2, d3, d4 = 64, 128, 256, 512
        
        # Downsampling factor = 8 (2 * 2 * 2)
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        
        # --- ENCODER (Backbone ResNet) ---
        self.encoder_net = nn.Sequential(
            # Layer iniziale per "agganciare" i colori
            nn.Conv2d(3, d1, kernel_size=7, stride=2, padding=3, bias=False), # res/2
            nn.GroupNorm(8, d1),
            nn.ReLU(inplace=True),
            
            StandardResidualBlock(d1, d2, stride=2),  # res/4
            StandardResidualBlock(d2, d3, stride=2),  # res/8
            StandardResidualBlock(d3, d4, stride=1),  # Bottleneck (mantiene res/8)
            
            # Compressione canali prima del Linear per non far esplodere i parametri
            nn.Conv2d(d4, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        flatten_dim = 64 * self.last_h * self.last_w
        
        # --- LATENT SPACE ---
        self.fc_encoder = nn.Linear(flatten_dim, model_dim)
        self.mean_predictor = nn.Linear(model_dim, latent_dim)
        self.logvar_predictor = nn.Linear(model_dim, latent_dim)

        # --- DECODER ---
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, flatten_dim),
            nn.ReLU(inplace=True)
        )

        self.decoder_net = nn.Sequential(
            nn.Conv2d(64, d4, kernel_size=1),
            StandardResidualBlock(d4, d3, stride=1),
            
            # Upsampling 1 (res/8 -> res/4)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(d3, d3, kernel_size=3, padding=1),
            StandardResidualBlock(d3, d2, stride=1),
            
            # Upsampling 2 (res/4 -> res/2)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(d2, d2, kernel_size=3, padding=1),
            StandardResidualBlock(d2, d1, stride=1),
            
            # Upsampling 3 (res/2 -> res/1)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(d1, d1, kernel_size=3, padding=1),
            
            nn.Conv2d(d1, 3, kernel_size=3, padding=1),
           
        )

    def encode(self, x):
        x = self.encoder_net(x)
        x = torch.flatten(x, 1)
        hidden = self.fc_encoder(x)
        return self.mean_predictor(hidden), self.logvar_predictor(hidden)

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 64, self.last_h, self.last_w)
        return self.decoder_net(z)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar