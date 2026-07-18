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


class ImpalaDecoderBlock(nn.Module):
    """
    Blocco di decodifica simmetrico a ImpalaBlock:
    1. Applica due blocchi residui per elaborare le feature alla risoluzione corrente.
    2. Utilizza una ConvTranspose2d per raddoppiare la risoluzione spaziale (l'opposto del MaxPool)
       e proiettare i canali verso la dimensione dello step successivo.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        # ConvTranspose2d con kernel 4, stride 2 e padding 1 raddoppia esattamente l'altezza e la larghezza
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.act(self.norm(self.up_conv(x)))
        return x


class ImpalaDecoder(L.LightningModule):
    def __init__(self, world_model_dim, model_dim, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        # Configurazione speculare dei canali basata su model_dim
        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        
        self.input_res = input_res
        self.model_dim = model_dim
        # Calcoliamo la dimensione spaziale compressa iniziale (es. 8x8 per un input 64x64)
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        flatten_dim = d3 * self.last_h * self.last_w
        
        # Proiezione lineare dallo spazio latente del World Model (world_model_dim)
        # allo spazio delle feature pronte per essere rimodellate in 3D
        self.decoder_input = nn.Sequential(
            nn.Linear(world_model_dim, flatten_dim),
            nn.SiLU()
        )

        self.decoder_net = nn.Sequential(
            # Up-sample 1: d3 -> d2 (es. da 8x8 a 16x16)
            ImpalaDecoderBlock(d3, d2),
            
            # Up-sample 2: d2 -> d1 (es. da 16x16 a 32x32)
            ImpalaDecoderBlock(d2, d1),
            
            # Up-sample 3: d1 -> 3 (es. da 32x32 a 64x64)
            ResidualBlock(d1),
            ResidualBlock(d1),
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
        )

    def decode(self, z):
        result = self.decoder_input(z)
        # Rimodelliamo il vettore piatto in un tensore 3D [Batch, Canali, Altezza, Larghezza]
        result = result.view(-1, self.model_dim, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        return self.decode(x)
    

class ResNetDecoder(L.LightningModule):
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
        )

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.model_dim, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        return self.decode(x)
    

class Decoder(L.LightningModule):
    def __init__(self, world_model_dim, model_dim, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        # Allineiamo il decoder al limite di 128 canali dell'encoder per la consistenza dimensionale
        self.cnn_dim = min(model_dim, 128)
        d1 = max(8, self.cnn_dim // 4)
        d2 = max(16, self.cnn_dim // 2)
        d3 = self.cnn_dim
        
        self.input_res = input_res
        self.model_dim = self.cnn_dim
        self.last_h = input_res[0] // 8
        self.last_w = input_res[1] // 8
        flatten_dim = d3 * self.last_h * self.last_w
        
        self.decoder_input = nn.Sequential(
            nn.Linear(world_model_dim, flatten_dim),
            nn.SiLU()
        )

        self.decoder_net = nn.Sequential(
            # Layer 1: Upsample (es. 20x15 -> 40x30 o 8x8 -> 16x16)
            nn.ConvTranspose2d(d3, d2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
            # Layer 2: Upsample (es. 40x30 -> 80x60 o 16x16 -> 32x32)
            nn.ConvTranspose2d(d2, d1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
            # Layer 3: Upsample finale (es. 80x60 -> 160x120 o 32x32 -> 64x64)
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
        )

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.model_dim, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        return self.decode(x)