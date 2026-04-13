import torch
from torch import nn
import lightning as L

class StandardResidualBlock(nn.Module):
    """
    Standard residual block for fine-features and resNET encoders and decoders
    """
    def __init__(self, in_channels, out_channels, stride=1, silu:bool=True):
        super().__init__()
       
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        
        
        # remove batch dependence 
        groups = min(8, out_channels)
        self.gn1 = nn.GroupNorm(groups, out_channels) 
        
        # performe convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        # activation
        self.act = nn.SiLU(inplace=True) if silu else nn.ReLU(inplace=True)

        # Shortcut per match delle dimensioni
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + residual)

class VAE(L.LightningModule):
    """Variational Autoencoder"""
    def __init__(self, latent_dim: int, model_dim: int = 256, input_res=(64, 64), silu:bool= True):
        super().__init__()
        self.save_hyperparameters()
        
       
        mult = max(1, model_dim // 16) 
        
        d1 = mult       # Es: 64
        d2 = mult * 2   # Es: 128
        d3 = mult * 4   # Es: 256
        d4 = mult * 8   # Es: 512
        
        # Downsampling factor = 8 (2 * 2 * 2)
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        
        self.act = nn.SiLU(inplace=True) if silu else nn.ReLU(inplace=True)
        
        # --- ENCODER ---
        self.encoder_net = nn.Sequential(
            
            # init block
            nn.Conv2d(3, d1, kernel_size=7, stride=2, padding=3, bias=False), # res/2
            nn.GroupNorm(min(8, d1), d1),
            self.act,
            
            # res blocks
            StandardResidualBlock(d1, d2, stride=2, silu=silu),  # res/4
            StandardResidualBlock(d2, d3, stride=2, silu=silu),  # res/8
            StandardResidualBlock(d3, d4, stride=1, silu=silu),  # Bottleneck (mantiene res/8)
            
            # out block
            nn.Conv2d(d4, d1, kernel_size=1),
            self.act
        )

        
        flatten_dim = d1 * self.last_h * self.last_w
        
        # --- LATENT SPACE ---
        self.fc_encoder = nn.Linear(flatten_dim, model_dim)
        self.mean_predictor = nn.Linear(model_dim, latent_dim)
        self.logvar_predictor = nn.Linear(model_dim, latent_dim)

        # --- DECODER ---
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            self.act,
            nn.Linear(model_dim, flatten_dim),
            self.act
        )

        self.decoder_net = nn.Sequential(
            nn.Conv2d(d1, d4, kernel_size=1),
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
            
            nn.Conv2d(d1, 3, kernel_size=3, padding=1)
        )

    def encode(self, x):
        """Encode image"""
        x = self.encoder_net(x)
        x = torch.flatten(x, 1)
        hidden = self.fc_encoder(x)
        return self.mean_predictor(hidden), self.logvar_predictor(hidden)

    def decode(self, z):
        """ decode features vector"""
        z = self.fc_decoder(z)
    
        d1 = max(1, self.hparams.model_dim // 16)
        z = z.view(-1, d1, self.last_h, self.last_w)
        return self.decoder_net(z)

    def reparameterize(self, mean, logvar):
        """reparametrization trick for ff optimization"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        """ forward step, full encode/decode step"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
    


if __name__ == "__main__":
    model = VAE(
        latent_dim=63, model_dim=128, input_res=(64,64), silu=False
    )

    data = torch.zeros( (8, 3, 64, 64) )
    print(data.shape)
    m, v = model.encode(data)

    print(f"encode dist: {m.shape}\n{v.shape}")

    z = model.decode(m)

    print(f"decode value: {z.shape}")

    model(data)

