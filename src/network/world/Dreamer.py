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
    

class PERAdapter(L.Callback):
    def __init__(self, buffer_train):
        super().__init__()
        self.buffer_train = buffer_train
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        index = outputs["episods_idx"]
        loss = outputs["episods_loss"]

        self.buffer_train.update(priority_idx=index, losses=loss)
        

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
        action_dim: int=16,
        att_dim:int=6,
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
        rssm_embed_dim = enc_dim + att_dim

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

        # encoders

        self.att_encoder = nn.Sequential(
            nn.Linear(att_dim, out_features=att_dim),
            nn.LayerNorm(att_dim*2),
            nn.SiLU()
        )

        
        self.episods_idx = []
        self.loss_episods = [] 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, att_user:torch.Tensor, mask:torch.Tensor|None=None) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs.shape

        # enc: [B,T,enc_dim]
        enc = self.encoder(obs)
        att_user = self.att_encoder(att_user)


        # -- fuse obs with user att

        # cat fusion
        enc = torch.cat([enc, att_user], dim=-1)
        #

        

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

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def get_loss_for_batch(self):
        """get loss for batch and index episods in batch"""
        idx_episods = torch.cat(self.episods_idx).numpy(True)
        loss_episods = torch.cat(self.loss_episods).numpy(True)
        self.episods_idx.clear()
        self.loss_episods.clear()
        return {
            "episods_idx":idx_episods, 
            "episods_loss":loss_episods
        }
    

    def validation_step(self, batch, batch_idx):
        # Estrazione dati (identica al training)
        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        cont = batch["done"]
        
        user_att = batch["user_att"]
        mask = batch["mask"]

        # Forward pass (assicurati che il modello sia in eval() mode per dropout/batchnorm)
        # In DreamerV3, il forward calcola la scomposizione dello stato latente
        out = self.forward(obs, actions, user_att, mask)

        # 1. Calcolo Loss di Ricostruzione e Predizione (Senza gradienti)
        # Usiamo 'none' per mantenere il controllo sulle dimensioni prima della media
        recon = F.binary_cross_entropy(out["x_hat"], obs, reduction="none").mean(dim=(2,3,4))
        reward_loss = F.mse_loss(out["r_hat"], rewards, reduction="none")
        cont_loss = F.binary_cross_entropy_with_logits(out["c_logits"], cont, reduction="none")
        
        L_pred_BT = recon + reward_loss + cont_loss 

        # 2. Calcolo KL Divergence
        # In validazione la KL ci dice quanto il "prior" (predizione) si allontana dal "posterior" (realtà)
        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]
        kl_val_BT = cat_kl_logits(q_logits, p_logits).sum(dim=-1)

        # 3. Calcolo Loss Totale (per logging)
        total_loss_BT = (
            self.hparams.beta_pred * L_pred_BT +
            self.hparams.beta_dyn * kl_val_BT # In val usiamo spesso la KL pura senza bilanciamento dyn/rep
        )

        # 4. Metriche Extra per Validazione
        # È utile monitorare l'accuratezza della predizione della reward (MAE)
        reward_mae = torch.abs(out["r_hat"] - rewards).mean()

        # Logghiamo i risultati per TensorBoard/WandB
        val_log = {
            "val_loss": total_loss_BT.mean(),
            "val_recon_loss": recon.mean(),
            "val_reward_loss": reward_loss.mean(),
            "val_reward_mae": reward_mae,
            "val_kl": kl_val_BT.mean()
        }
        
        self.log_dict(val_log, on_step=False, on_epoch=True, prog_bar=True)
        

        # Nota: non aggiorniamo self.loss_episods qui per non sporcare 
        # le statistiche di training con i dati di validazione.
        return val_log
    
    def training_step(self, batch, batch_idx):
        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        cont = batch["done"]
        
        idx_episode = batch["priority_idx"]
        user_att = batch["user_att"]
        mask = batch["mask"]

        out = self.forward(obs, actions, user_att, mask)

        # Ricostruzione immagini
        recon = F.binary_cross_entropy(out["x_hat"], obs, reduction="none").mean(dim=(2,3,4))  # [B,T]
        reward_loss = F.mse_loss(out["r_hat"], rewards, reduction="none")                      # [B,T]
        cont_loss = F.binary_cross_entropy_with_logits(out["c_logits"], cont, reduction="none")# [B,T]
        
        # --- 1. Calcolo componenti Predizione [B, T] ---
        
        L_pred_BT = recon + reward_loss + cont_loss 

        # --- 2. KL components ---
        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]

        
        kl_dyn_BT = cat_kl_logits(q_logits.detach(), p_logits).sum(dim=-1)
        kl_rep_BT = cat_kl_logits(q_logits, p_logits.detach()).sum(dim=-1)

        # clamping 
        free = self.hparams.free_nats
        kl_dyn_BT = torch.clamp(kl_dyn_BT, min=free)
        kl_rep_BT = torch.clamp(kl_rep_BT, min=free)

        # --- 3. total loss
        total_loss_BT = (
            self.hparams.beta_pred * L_pred_BT +
            self.hparams.beta_dyn * kl_dyn_BT +
            self.hparams.beta_rep * kl_rep_BT
        )

        # --- 4. results ---

        loss_per_batch = total_loss_BT.mean(dim=1) 
        self.episods_idx.append(idx_episode)
        self.loss_episods.append(loss_per_batch)

        r =  {"loss":loss_per_batch.mean()} | self.get_loss_for_batch()
        return r