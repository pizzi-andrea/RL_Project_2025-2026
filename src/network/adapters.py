import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class VAEAdapter(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    

class PERAdapter(L.Callback):
    def __init__(self, buffer):
        super().__init__()
        self.buffer_train = buffer
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        index = outputs["episods_idx"]
        loss = outputs["episods_loss"]
        
        self.buffer_train.update(priority_idx=index, losses=loss)
        
class VAEEncoderAdapter(VAEAdapter):
    """
    Prende immagini [B,T,3,H,W] o [B,3,H,W] e ritorna enc_x.
    Qui enc_x = mean del VAE (deterministico).
    """
    def __init__(self, vae, out_dim:int):
        super().__init__(vae)
        self.out_dim = out_dim
        self.proj = None
        if out_dim != vae.hparams.latent_dim:
            self.proj = nn.Sequential(
                nn.Linear(vae.hparams.latent_dim, out_dim),
                nn.RMSNorm(out_dim),
                nn.SiLU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] oppure [B,C,H,W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            
            
            x = x.reshape(B * T, C, H, W)
            mean, _ = self.vae.encode(x)  # mean: [B*T, latent_dim]
            if self.proj is not None:
                mean = self.proj(mean)
            return mean.view(B, T, -1)
        else:
            mean, _ = self.vae.encode(x)
            if self.proj is not None:
                mean = self.proj(mean)
            return mean



class VAEDecoderAdapter(VAEAdapter):
    def __init__(self, vae, feat_dim:int, decode_shape:tuple[int,int, int] =(3, 120,160)):
        super().__init__(vae)
        self.decode_shape = decode_shape
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, self.vae.hparams.latent_dim),
            nn.RMSNorm(self.vae.hparams.latent_dim),
            nn.SiLU()
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            B, T, D = feat.shape
            # Appiattisci B e T
            z = feat.view(B * T, D) 
            z = self.feat_proj(z)
            
            # Decodifica standard (output sarà [B*T, 3, 240, 320])
            imgs = self.vae.decode(z) 
            
            # Reshape manuale alla fine
            return imgs.view(B, T, self.decode_shape[0], self.decode_shape[1], self.decode_shape[2])
        else:
            z = self.feat_proj(feat)
            return self.vae.decode(z)


