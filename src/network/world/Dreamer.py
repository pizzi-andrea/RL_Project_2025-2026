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

# RNN network
from ..dreamer.rssm import RSSM

from .utils import *
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
        index = outputs["episods_idx"].cpu().detach().numpy().squeeze(-1)
        loss = outputs["episods_loss"].cpu().detach().numpy()
        
        self.buffer_train.update(priority_idx=index, losses=loss)
        

class VAEEncoderAdapter(VAEAdapter):
    """
    Prende immagini [B,T,3,H,W] o [B,3,H,W] e ritorna enc_x.
    Qui enc_x = mean del VAE (deterministico).
    """
    def __init__(self, vae: VAE, out_dim:int):
        super().__init__(vae)
        self.out_dim = out_dim
        self.proj = None
        if out_dim != vae.hparams.latent_dim:
            self.proj = nn.Sequential(
                nn.Linear(vae.hparams.latent_dim, out_dim),
                nn.LayerNorm(out_dim),
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
    def __init__(self, vae: VAE, feat_dim:int, decode_shape:tuple[int,int, int] =(3, 120,160)):
        super().__init__(vae)
        self.decode_shape = decode_shape
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, self.vae.hparams.latent_dim),
            nn.LayerNorm(self.vae.hparams.latent_dim),
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



# -------------------------
# World Model LightningModule - need use REP buffer - remove 
# -------------------------
class Dreamer(L.LightningModule):
    def __init__(
        self,
        action_dim: int=16,
        att_dim:int=6,
        internal_hidden_dim = 1024,
        rssm_hidden_dim = 1024,
        h_dim: int = 512,
        stoch_latents: int = 32,
        stoch_classes: int = 32,
        decode_shape: tuple[int,int,int] = (3, 120,160),
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

        
        self.rssm = RSSM(
            action_dim=action_dim,
            deter_dim=h_dim,
            obs_dim=internal_hidden_dim,
            stoch_latents=stoch_latents,
            stoch_classes=stoch_classes,
            embed_dim=rssm_hidden_dim,
            unimix=unimix,
        )

        rssm_feat_dim = h_dim + (stoch_latents * stoch_classes) # h_dim + L*K
        self.internal_hidden_dim  = internal_hidden_dim
        

        
            
        if isinstance(vae, str):
            self.vae = VAE.load_from_checkpoint(vae, map_location='cpu')
        elif isinstance(vae, VAE):
            self.vae = vae
        elif not vae:
            self.vae = VAE(latent_dim=internal_hidden_dim)
        else:
            assert "No VAE instance"

        # -- encoders
        # encoder: enc_x = mean (dimensione latent_dim o proiettata a enc_dim)
        self.encoder = VAEEncoderAdapter(self.vae, out_dim=internal_hidden_dim)

        # decoder: usa decoder VAE, aggiungendo solo feat->latent
        self.decoder = VAEDecoderAdapter(self.vae, feat_dim=internal_hidden_dim, decode_shape=decode_shape)

        self.att_encoder = nn.Sequential(
            nn.Linear(att_dim, out_features=internal_hidden_dim),
            nn.LayerNorm(internal_hidden_dim),
            nn.SiLU()
        )

        self.encoder_proj = nn.Sequential(
            nn.Linear(2*internal_hidden_dim, internal_hidden_dim),
            nn.LayerNorm(internal_hidden_dim),
            nn.SiLU()
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(rssm_feat_dim, internal_hidden_dim),
            nn.LayerNorm(internal_hidden_dim),
            nn.SiLU()
        )


        # -- controllers

        # reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.LayerNorm(internal_hidden_dim),
            nn.SiLU(),
            nn.Linear(internal_hidden_dim, 1),
        )

        # continue predictor
        self.cont_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.LayerNorm(internal_hidden_dim),
            nn.SiLU(),
            nn.Linear(internal_hidden_dim, 1),
        )

        # attribute predictor
        self.att_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.LayerNorm(internal_hidden_dim),
            nn.SiLU(),
            nn.Linear(internal_hidden_dim, att_dim)
        )

        # -- encoders

        
    
    
    def compute_loss(self, out, batch):
        """
        Compute RSSM loss
        """
        # -
        obs = batch["obs"]           # [B, T, C, H, W]
        rewards = batch["reward"]    # [B, T]
        cont = batch["done"]         # [B, T]
        user_att = batch["user_att"] # [B, T, AttDim]
        mask = batch["mask"]         # [B, T]
        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]

        obs_hat = out["x_hat"]
        rewards_hat = out["r_hat"]
        cont_hat = out["c_logits"]
        user_att_hat = out["att_hat"]
        
        loss_pred, loss_pred_t = loss_prediction(obs_hat, rewards_hat, cont_hat, user_att_hat, obs, rewards,cont, user_att, mask=mask, beta=self.hparams.beta_pred)
        loss_dy, loss_dy_t, loss_dy_raw = loss_dynamic(q_logits, p_logits, mask=mask, beta=self.hparams.beta_dyn)
        loss_rep, loss_rep_t, loss_rep_raw = loss_representation(q_logits, p_logits, mask=mask, beta=self.hparams.beta_rep)


        
        loss = loss_pred + loss_dy + loss_rep
        loss_t = self.hparams.beta_pred*loss_pred_t + self.hparams.beta_dyn*loss_dy_t + self.hparams.beta_rep*loss_rep_t


        # 
        
        components = {
            "prediction_loss": loss_pred.detach().cpu().item(),
            "dynamic_loss": loss_dy_raw,
            "representation_loss": loss_rep_raw,
           
        }

        return loss, components, loss_t
    
    def setup(self, stage):
        self.unfreeze()
    
    def freeze_model(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_model(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def configure_optimizers(self):
        
        return torch.optim.AdamW(self.parameters(), self.hparams.lr)
    
    def encode(self, obs, user_att):
        enc = self.encoder(obs)
        user_att = symlog(user_att)
        att_user = self.att_encoder(user_att)
        
        # -- fuse obs with user att, use simple cat fusion
        enc = torch.cat([enc, att_user], dim=-1)
        embs_encode_state = self.encoder_proj(enc)
        return embs_encode_state

    def predict_reward(self, feat):
        return self.reward_head(feat).squeeze(-1)
    
    def predict_continue(self, feat):
        return self.cont_head(feat).squeeze(-1)
    
    def predict_obs(self, feat):
        return self.decoder(feat) 
    
    def predict_user_att(self, feat):
        return self.att_head(feat)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor, att_user:torch.Tensor, mask:torch.Tensor|None=None) -> Dict[str, torch.Tensor]:
        B, T, _, _, _ = obs.shape

        
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
        feat = self.feat_proj(feat)
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
        

        # Logging Validation
        loss, components, _ = self.compute_loss(out, batch)
        components["validation_loss"] = loss
        self.log_dict(components, prog_bar=True, on_step=True, on_epoch=True)
        
        # Gestione Immagini (solo batch 0)
        
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
        
    
    def training_step(self, batch, batch_idx):
        # 1. Forward
        priority_idx = batch["priority_idx"].detach().cpu()
        out = self.forward(batch["obs"], batch["action"], batch["user_att"], batch["mask"])
        
        # 2. compute loss
        loss, components, loss_t = self.compute_loss(out, batch)
        components["train_loss"] = loss
        
        
        self.log_dict(components, prog_bar=True, on_step=True, on_epoch=True)

        # 4. Return per PER e Ottimizzatore
        return {
            "loss": loss,
            "episods_idx": priority_idx,
            "episods_loss": loss_t.detach().cpu().mean(dim=-1)
        }