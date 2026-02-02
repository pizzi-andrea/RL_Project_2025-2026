import torchvision
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
from ..dreamer.rssm import RSSM

from .utils import cat_kl_logits
# -------------------------
# VAE adaptation
# -------------------------

class VAEAdapter(nn.Module):
    def __init__(self, vae: VAE):
        super().__init__()
        self.vae = vae
    

class PERAdapter(L.Callback):
    def __init__(self, buffer):
        super().__init__()
        self.buffer_train = buffer
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        index = outputs["episods_idx"].squeeze(-1).cpu().detach().numpy()
        loss = outputs["episods_loss"].cpu().detach().numpy()

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
    def __init__(self, vae: VAE, feat_dim: int):
        super().__init__(vae)
        latent_dim = vae.hparams.latent_dim
        
        self.feat_to_latent = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.SiLU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            B, T, D = feat.shape
            # Appiattisci B e T
            z = self.feat_to_latent(feat.view(B * T, D)) 
            
            # Decodifica standard (output sarà [B*T, 3, 240, 320])
            imgs = self.vae.decode(z) 
            
            # Reshape manuale alla fine
            return imgs.view(B, T, 3, 240, 320)
        else:
            z = self.feat_to_latent(feat)
            return self.vae.decode(z)



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
        vae: Optional[str]|VAE|None = None,
        
        # se vuoi che enc_x sia esattamente latent_dim del VAE, imposta enc_dim=vae_latent_dim
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])

        # RSSM (embed_dim deve matchare la dimensione di enc_x)
        # Se usi VAE, il modo più semplice è enc_dim == vae_latent_dim
        rssm_embed_dim = enc_dim + att_dim
        self.rssm_embed_dim = rssm_embed_dim

        self.rssm = RSSM(
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_latents=stoch_latents,
            stoch_classes=stoch_classes,
            embed_dim=rssm_embed_dim,
            unimix=unimix,
        )

        feat_dim = deter_dim + stoch_latents * stoch_classes

        
            
        if isinstance(vae, str):
            self.vae = VAE.load_from_checkpoint(vae, map_location='cpu')
        elif isinstance(vae, VAE):
            self.vae = vae
        elif not vae:
            self.vae = VAE(latent_dim=128)
        else:
            assert "No VAE instance"

        # encoder: enc_x = mean (dimensione latent_dim o proiettata a enc_dim)
        self.encoder = VAEEncoderAdapter(self.vae, out_dim=enc_dim)

        # decoder: usa decoder VAE, aggiungendo solo feat->latent
        self.decoder = VAEDecoderAdapter(self.vae, feat_dim=feat_dim)

        # con VAE l'immagine è 240x320 (come nel vostro VAE)
        self.obs_h = 240
        self.obs_w = 320
        

        # -- controller
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

        self.att_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, att_dim)
        )

        # -- encoders

        self.att_encoder = nn.Sequential(
            nn.Linear(att_dim, out_features=att_dim),
            nn.LayerNorm(att_dim),
            nn.SiLU()
        )
    def symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    def compute_loss(self, out, batch):
        """
        Calcola solo i valori delle loss. Niente logging, niente .detach().
        Ritorna tensori pronti per la backward pass.
        """
        # --- 1. Unpack ---
        obs = batch["obs"]           # [B, T, C, H, W]
        rewards = batch["reward"]    # [B, T]
        cont = batch["done"]         # [B, T]
        user_att = batch["user_att"] # [B, T, AttDim]
        mask = batch["mask"]         # [B, T]
        
        #--- Reconstruction Loss (Image)
        
        recon_loss_pixel = F.binary_cross_entropy_with_logits(out["x_hat"], obs, reduction="none")
        recon_loss = recon_loss_pixel.mean(dim=(2, 3, 4)) # normalize loss

        #--- Prediction Losses

        # first normalize
        out["r_hat"] = torch.sign(out["r_hat"]) * torch.log(1 + torch.abs(out["r_hat"]))
        rewards = torch.sign(rewards) * torch.log(1 + torch.abs(rewards))

        reward_loss = F.mse_loss(out["r_hat"], rewards, reduction="none")              # [B, T]
        
        user_att = self.symlog(user_att)
        out["att_hat"] = self.symlog(out["att_hat"])
        att_loss = F.mse_loss(out["att_hat"], user_att, reduction="none").sum(dim=-1)  # [B, T]
        

        cont_loss = F.binary_cross_entropy_with_logits(out["c_logits"], cont, reduction="none") # [B, T]

        L_pred = recon_loss + reward_loss + cont_loss + att_loss

        #--- KL Divergence 
        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]

        # Sum su ultima dim (categorie)
        kl_dyn_raw = cat_kl_logits(q_logits, p_logits).sum(dim=-1)        # [B, T]
        kl_rep_raw = cat_kl_logits(q_logits.detach(), p_logits).sum(dim=-1) # [B, T] - nota il detach su q qui per la rappresentazione

        # Free Nats Clamping
        free = torch.tensor(self.hparams.free_nats, device=self.device)
        kl_dyn = torch.max(kl_dyn_raw, free)
        kl_rep = torch.max(kl_rep_raw, free)

        # --- 5. Weighted Loss (con Masking) ---
        weighted_loss = (
            self.hparams.beta_pred * L_pred +
            self.hparams.beta_dyn * kl_dyn +
            self.hparams.beta_rep * kl_rep
        ) # [B, T]

        # Applica maschera (zero loss su padding)
        weighted_loss = weighted_loss * mask
        
        # Normalizzazione: Somma totale / numero elementi validi
        denom = mask.sum().clamp(min=1.0)
        final_loss = weighted_loss.sum() / denom

        # --- 6. Loss per Batch (per PER) ---
        
        batch_denom = mask.sum(dim=1).clamp(min=1.0)
        loss_per_batch = weighted_loss.sum(dim=1) / batch_denom

        # --- 7. Packing Components ---
        
        components = {
            "recon_loss": (recon_loss * mask).sum() / denom,
            "reward_loss": (reward_loss * mask).sum() / denom,
            "cont_loss": (cont_loss * mask).sum() / denom,
            "att_loss": (att_loss * mask).sum() / denom,
            "kl_dyn": (kl_dyn_raw * mask).sum() / denom,
            "kl_rep": (kl_rep_raw * mask).sum() / denom,
        }

        return final_loss, components, loss_per_batch
    def freeze_model(self):
        for p in self.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def encode(self, obs, user_att):
        enc = self.encoder(obs)
        att_user = self.att_encoder(user_att)
        
        # -- fuse obs with user att, use simple cat fusion
        enc = torch.cat([enc, att_user], dim=-1)
        return enc

    def predict_reward(self, feat):
        return self.reward_head(feat).squeeze(-1)
    def predict_continue(self, feat):
        return self.cont_head(feat).squeeze(-1)
    def predict_obs(self, feat):
        
        return self.decoder(feat) 
    
    def predict_user_att(self, feat):
        return self.att_head(feat)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor, att_user:torch.Tensor, mask:torch.Tensor|None=None) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs.shape

        
        # -- init memory
        state = self.rssm.init_state(B, device=obs.device)

        # -- store temp data
        post_h, post_z, post_logits, prior_logits = [], [], [], []
        
        for t in range(T):
            a_t = actions[:, t]
            att = att_user[:, t]
            m = mask[:,t]
            #print(f"att shape: {att.shape}")
            x_t = self.encode(obs[:, t], att)  # encode observation + user attributes
            state, plogits = self.rssm.observe_step(state, a_t, x_t, m)
            post_h.append(state.h)
            post_z.append(state.z)
            post_logits.append(state.z_logits)
            prior_logits.append(plogits)

        post_h = torch.stack(post_h, dim=1)            # [B,T,H]
        post_z = torch.stack(post_z, dim=1)            # [B,T,L*K]
        post_logits = torch.stack(post_logits, dim=1)  # [B,T,L,K]
        
        prior_logits = torch.stack(prior_logits, dim=1)# [B,T,L,K]

        feat = torch.cat([post_h, post_z], dim=-1)     # [B,T,feat_dim]
        
        x_hat = self.predict_obs(feat)                    # [B,T,3,H,W] (H,W dipende da VAE o conv)
        r_hat = self.predict_reward(feat)    # [B,T]
        att_hat = self.predict_user_att(feat)
        c_logits = self.predict_continue(feat)   # [B,T]


        return {
            "x_hat": x_hat,
            "r_hat": r_hat,
            "c_logits": c_logits,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
            "att_hat": att_hat
        }

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def _get_loss_for_batch(self, episods_idx, loss_episods):
        """get loss for batch and index episods in batch"""
        idx_episods = episods_idx.detach().cpu().numpy()
        loss_episods = loss_episods.detach().cpu().numpy()
        
        return {
            "episods_idx":idx_episods, 
            "episods_loss":loss_episods
        }
    
    

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch["obs"], batch["action"], batch["user_att"], batch["mask"])
        
        loss, components, _ = self.compute_loss(out, batch)

        # Logging Validation
        log_dict = {f"val_{k}": v.detach() for k, v in components.items()}
        log_dict["val_loss"] = loss.detach()
        self.log_dict(log_dict, prog_bar=True)
        
        # Gestione Immagini (solo batch 0)
        if batch_idx == 0:
            self._log_validation_images(batch["obs"], out["x_hat"])
            
    def _log_validation_images(self, true_obs, pred_logits):
        # Helper per pulire validation_step
        # pred_logits sono logits -> sigmoid per visualizzare
        recon_img = torch.sigmoid(pred_logits[:, 0]) 
        true_img = true_obs[:, 0]
        # Prendi max 8 immagini
        n = min(true_img.shape[0], 8)
        grid = torchvision.utils.make_grid(torch.cat([true_img[:n], recon_img[:n]]), nrow=n)
        self.logger.experiment.add_image('val_recon', grid, self.current_epoch)
        

        
        return val_log
    
    def training_step(self, batch, batch_idx):
        # 1. Forward
        out = self.forward(batch["obs"], batch["action"], batch["user_att"], batch["mask"])

        # 2. Calcolo Loss (Puro)
        loss, components, loss_per_batch = self.compute_loss(out, batch)

        # 3. Preparazione Log (Logica di UI/Monitoraggio)
        # Stacchiamo i gradienti qui per il logger
        log_dict = {f"train_{k}": v.detach() for k, v in components.items()}
        log_dict["train_loss"] = loss.detach()
        
        # Aggiungi metriche extra se vuoi (es. reward media)
        mask = batch["mask"]
        denom = mask.sum().clamp(min=1.0)
        log_dict["train_avg_reward"] = (out["r_hat"].detach() * mask).sum() / denom

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        # 4. Return per PER e Ottimizzatore
        return {
            "loss": loss,
            "episods_idx": batch["priority_idx"],
            "episods_loss": loss_per_batch.detach()
        }