from torch import nn
import lightning as L


class VAE(L.LightningModule):
    def __init__(self, 
    
    # dimension of feactures vector exstract by encoder and expected by decoder
    latent_dim: int, 
    # regulate internal filter dimension
    model_dim: int, 
    
    # image input dimension
    input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        d4 = model_dim * 2
        self.input_res = input_res

        self.last_h = input_res[0] // 16
        self.last_w = input_res[1] // 16
        flatten_dim = d4 * self.last_h * self.last_w

        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, d1, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d1),
            nn.SiLU(),
            nn.Conv2d(d1, d2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d2),
            nn.SiLU(),
            nn.Conv2d(d2, d3, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d3),
            nn.SiLU(),
            nn.Conv2d(d3, d4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d4),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim),
        )

        

    def encode(self, x):
        # x: [B, 3, H, W] → (embed: [B, latent_dim], None)
        # Ritorna tupla per compatibilità con VAEEncoderAdapter (mean, _)
        return self.encoder_net(x)

    def decode(self, z):
        # z: [B, latent_dim] → [B, 3, H, W] logits
       
        result = self.decoder_input(z)
        result = result.view(-1, self.hparams.model_dim * 2, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        embed = self.encode(x)
        return embed
    

class Decoder(L.LightningModule):

    def __init__(self, world_model_dim, model_dim, input_res=(64, 64)):
        super().__init__()
        self.save_hyperparameters()

        d1 = max(1, model_dim // 4)
        d2 = max(1, model_dim // 2)
        d3 = model_dim
        d4 = model_dim * 2
        self.input_res = input_res
        self.model_dim = model_dim

        self.last_h = input_res[0] // 16
        self.last_w = input_res[1] // 16
        flatten_dim = d4 * self.last_h * self.last_w
        self.decoder_input = nn.Linear(world_model_dim, flatten_dim)

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(d4, d3, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d3),
            nn.SiLU(),
            nn.ConvTranspose2d(d3, d2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d2),
            nn.SiLU(),
            nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=1),
            nn.InstanceNorm2d(d1),
            nn.SiLU(),
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
        )


    def forward(self, x):
        embed  = self.decode(x)
        return embed

    def decode(self, z):
        result = self.decoder_input(z)
        
        
        result = result.view(-1, self.model_dim * 2, self.last_h, self.last_w)
        return self.decoder_net(result)