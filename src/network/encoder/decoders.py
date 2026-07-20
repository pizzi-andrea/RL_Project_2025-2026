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
 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)

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
         
            ImpalaDecoderBlock(d3, d2),
            
           
            ImpalaDecoderBlock(d2, d1),
         
            ResidualBlock(d1),
            ResidualBlock(d1),
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
        )

    def decode(self, z):
        result = self.decoder_input(z)
     
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
           
            ResidualBlock(d3),
            nn.ConvTranspose2d(d3, d2, 4, stride=2, padding=1),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
           
            ResidualBlock(d2),
            nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=1),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
  
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
           
            nn.ConvTranspose2d(d3, d2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d2),
            nn.SiLU(),
            
  
            nn.ConvTranspose2d(d2, d1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, d1),
            nn.SiLU(),
            
  
            nn.ConvTranspose2d(d1, 3, 4, stride=2, padding=1),
        )

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.model_dim, self.last_h, self.last_w)
        return self.decoder_net(result)

    def forward(self, x):
        return self.decode(x)