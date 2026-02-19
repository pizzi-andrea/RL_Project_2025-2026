import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
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
        
        update_episods, new_priority, count_sample = self.buffer_train.update(priority_idx=index, losses=loss)
        
        if batch_idx == 0:     
            trainer.print("new priority...")
            trainer.print(f"[PER Callback]:")
            trainer.print(f"Sample episods (last): {update_episods}")
            trainer.print(f"Sample Priorities (last): {new_priority}")
            trainer.print(f"Sample Counts (last): {count_sample}")
       


        
class VAEEncoderAdapter(VAEAdapter):
    """
    Prende immagini [B,T,3,H,W] o [B,3,H,W] e ritorna enc_x.
    Qui enc_x = mean del VAE (deterministico).
    Gestisce la proiezione se la dimensione del latent del VAE differisce dalla dimensione richiesta (es. model_dim).
    """
    def __init__(self, vae, out_dim:int):
        super().__init__(vae)
        self.out_dim = out_dim
        self.proj = None
        
        # Se la dimensione richiesta (es. model_dim del Dreamer) è diversa dal latent del VAE,
        # proiettiamo l'output.
        if out_dim != vae.hparams.latent_dim:
            print("Enable proj on VAE Encoder")
            self.proj = nn.Sequential(
                nn.Linear(vae.hparams.latent_dim, out_dim)
            )
            # Inizializzazione coerente con il nuovo VAE
            self._init_weights(self.proj)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] oppure [B,C,H,W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            
            x = x.reshape(B * T, C, H, W)
            # Nota: il VAE nuovo esegue la normalizzazione in forward(), ma encode() potrebbe non farlo.
            # Si assume che 'x' sia già pre-processato o che encode del VAE gestisca il tipo dati.
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
    """
    Adatta l'input feature (stato RSSM) al latent del VAE per la decodifica.
    """
    def __init__(self, vae, feat_dim:int, decode_shape:tuple[int,int, int] =(3, 120,160)):
        super().__init__(vae)
        self.decode_shape = decode_shape
        
        # Proietta le feature dello stato (h + z) verso la dimensione latent del VAE
        if feat_dim != self.vae.hparams.latent_dim:
            print("Enable proj on VAE Decoder")
            self.feat_proj = nn.Sequential(
                nn.Linear(feat_dim, self.vae.hparams.latent_dim)
            )
        else:
            self.feat_proj = nn.Identity()
        
        # Inizializzazione coerente con il nuovo VAE
        self._init_weights(self.feat_proj)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            B, T, D = feat.shape
            # Appiattisci B e T
            z = feat.view(B * T, D) 
            z = self.feat_proj(z)
            
            # Decodifica standard (output sarà [B*T, C, H, W])
            imgs = self.vae.decode(z) 
            
            # Reshape manuale alla fine per garantire [B, T, C, H, W]
            return imgs.view(B, T, self.decode_shape[0], self.decode_shape[1], self.decode_shape[2])
        else:
            z = self.feat_proj(feat)
            return self.vae.decode(z)