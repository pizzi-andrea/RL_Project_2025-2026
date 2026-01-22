from ..encoder.VAE import VAE
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Optional

# Vision Encoders
from ..encoder.VAE import VAE
from ..encoder.convEncoder import ConvEncoder, ConvDecoder

# RNN network
from ..RSSM.rssm import RSSM

from utils import cat_kl_logits
# -------------------------
# VAE adaptation
# -------------------------

class VAEAdapter(nn.Module):
    def __init__(self, vae: VAE):
        super().__init__()
        self.vae = vae
    
    

class VAEEncoderAdapter(VAEAdapter):
    """
    Prende immagini [B,T,3,H,W] o [B,3,H,W] e ritorna enc_x.
    Qui enc_x = mean del VAE (deterministico).
    """
    def __init__(self, vae: VAE, out_dim: Optional[int] = None):
        super().__init__(vae)
        self.out_dim = out_dim
        self.proj = None
        if out_dim is not None and out_dim != vae.hparams.latent_dim:
            self.proj = nn.Linear(vae.hparams.latent_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] oppure [B,C,H,W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            _, mean, _ = self.vae.encode(x)  # mean: [B*T, latent_dim]
            if self.proj is not None:
                mean = self.proj(mean)
            return mean.view(B, T, -1)
        else:
            _, mean, _ = self.vae.encode(x)
            if self.proj is not None:
                mean = self.proj(mean)
            return mean


class VAEDecoderAdapter(VAEAdapter):
    """
    Prende feat = [h,z] e produce immagine usando il decoder del VAE.
    Aggiunge un layer nuovo che mappa feat_dim -> latent_dim.
    """
    def __init__(self, vae: VAE, feat_dim: int):
        super().__init__(vae)
        latent_dim = vae.hparams.latent_dim
        self.feat_to_latent = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.SiLU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,T,D] oppure [B,D]
        if feat.dim() == 3:
            B, T, D = feat.shape
            z = self.feat_to_latent(feat.view(B * T, D))          # [B*T, latent_dim]
            
            z_j = self.vae.z_project(z).view(B * T, 256, 15, 20)  # usa pesi pretrainati
            x = self.vae.decoder(z_j).view(B, T, 3, 240, 320)     # output VAE: 320x240
            return x
        else:
            
            return self.vae.decode(z, (-1, 256, 15, 20))



# -------------------------
# World Model LightningModule - need use REP buffer - remove 
# -------------------------
class Dreamer(L.LightningModule):
    def __init__(
        self,
        action_dim: int,
        image_channels: int = 3,
        deter_dim: int = 1024,
        stoch_latents: int = 32,
        stoch_classes: int = 32,
        enc_dim: int = 1024,
        lr: float = 3e-4,
        beta_pred: float = 1.0,
        beta_dyn: float = 1.0,
        beta_rep: float = 0.1,
        free_nats: float = 1.0,
        unimix: float = 0.01,
        # --- nuove opzioni VAE ---
        
        vae_ckpt_path: Optional[str] = None,
        vae_latent_dim: int = 128,
        freeze_vae: bool = True,
        # se vuoi che enc_x sia esattamente latent_dim del VAE, imposta enc_dim=vae_latent_dim
    ):
        super().__init__()
        self.save_hyperparameters()

        # RSSM (embed_dim deve matchare la dimensione di enc_x)
        # Se usi VAE, il modo più semplice è enc_dim == vae_latent_dim
        rssm_embed_dim = enc_dim

        self.rssm = RSSM(
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_latents=stoch_latents,
            stoch_classes=stoch_classes,
            embed_dim=rssm_embed_dim,
            unimix=unimix,
        )

        feat_dim = deter_dim + stoch_latents * stoch_classes

        if vae_ckpt_path:
            assert vae_ckpt_path is not None, "Se use_pretrained_vae=True devi passare vae_ckpt_path"
            self.vae = VAE.load_from_checkpoint(vae_ckpt_path, map_location='cpu')

            if freeze_vae:
                for p in self.vae.parameters():
                    p.requires_grad = False

            # encoder: enc_x = mean (dimensione latent_dim o proiettata a enc_dim)
            self.encoder = VAEEncoderAdapter(self.vae, out_dim=enc_dim)

            # decoder: usa decoder VAE, aggiungendo solo feat->latent
            self.decoder = VAEDecoderAdapter(self.vae, feat_dim=feat_dim)

            # con VAE l'immagine è 240x320 (come nel vostro VAE)
            self.obs_h = 240
            self.obs_w = 320
        else:
            self.encoder = ConvEncoder(in_ch=image_channels, out_dim=enc_dim)
            self.decoder = ConvDecoder(in_dim=feat_dim, out_ch=image_channels)
            self.obs_h = 240
            self.obs_w = 320

        # controller
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, deter_dim),
            nn.SiLU(),
            nn.Linear(deter_dim, 1),
        )
        self.cont_head = nn.Sequential(
            nn.Linear(feat_dim, deter_dim),
            nn.SiLU(),
            nn.Linear(deter_dim, 1),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs.shape

        # enc: [B,T,enc_dim]
        enc = self.encoder(obs)

        state = self.rssm.init_state(B, device=obs.device)

        post_h, post_z, post_logits, prior_logits = [], [], [], []

        for t in range(T):
            a_t = actions[:, t]
            x_t = enc[:, t]  # enc_x
            state, plogits = self.rssm.observe_step(state, a_t, x_t)
            post_h.append(state.h)
            post_z.append(state.z)
            post_logits.append(state.z_logits)
            prior_logits.append(plogits)

        post_h = torch.stack(post_h, dim=1)            # [B,T,H]
        post_z = torch.stack(post_z, dim=1)            # [B,T,L*K]
        post_logits = torch.stack(post_logits, dim=1)  # [B,T,L,K]
        
        prior_logits = torch.stack(prior_logits, dim=1)# [B,T,L,K]

        feat = torch.cat([post_h, post_z], dim=-1)     # [B,T,feat_dim]
        x_hat = self.decoder(feat)                    # [B,T,3,H,W] (H,W dipende da VAE o conv)
        r_hat = self.reward_head(feat).squeeze(-1)    # [B,T]
        c_logits = self.cont_head(feat).squeeze(-1)   # [B,T]

        return {
            "x_hat": x_hat,
            "r_hat": r_hat,
            "c_logits": c_logits,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
        }

    def training_step(self, batch, batch_idx):
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        cont = batch["cont"]

        out = self.forward(obs, actions)

        # Ricostruzione immagini
        recon = F.binary_cross_entropy(out["x_hat"], obs, reduction="none").mean(dim=(2,3,4))  # [B,T]
        reward_loss = F.mse_loss(out["r_hat"], rewards, reduction="none")                      # [B,T]
        cont_loss = F.binary_cross_entropy_with_logits(out["c_logits"], cont, reduction="none")# [B,T]
        L_pred = (recon + reward_loss + cont_loss).mean()

        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]

        kl_dyn = cat_kl_logits(q_logits.detach(), p_logits).sum(dim=-1)  # [B,T]
        kl_rep = cat_kl_logits(q_logits, p_logits.detach()).sum(dim=-1)  # [B,T]

        free = self.hparams.free_nats
        kl_dyn = torch.clamp(kl_dyn, min=free).mean()
        kl_rep = torch.clamp(kl_rep, min=free).mean()

        loss = (
            self.hparams.beta_pred * L_pred +
            self.hparams.beta_dyn * kl_dyn +
            self.hparams.beta_rep * kl_rep
        )

        self.log_dict({
            "wm/loss": loss,
            "wm/pred": L_pred,
            "wm/kl_dyn": kl_dyn,
            "wm/kl_rep": kl_rep,
            "wm/recon": recon.mean(),
            "wm/r_loss": reward_loss.mean(),
            "wm/c_loss": cont_loss.mean(),
        }, prog_bar=True)

        return loss