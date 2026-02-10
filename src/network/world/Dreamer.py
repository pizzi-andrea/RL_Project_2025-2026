import torchvision
from ..encoder.VAE import VAE
from typing import Dict
from timm.optim.laprop import LaProp
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
from ..adapters import *
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
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU()
        )

        self.encoder_proj = nn.Sequential(
            nn.Linear(2*internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU()
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(rssm_feat_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU()
        )


        # -- controllers

        # reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
            nn.Linear(internal_hidden_dim, 1),
        )

        # continue predictor
        self.cont_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
            nn.Linear(internal_hidden_dim, 1),
        )

        # attribute predictor
        self.att_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
            nn.Linear(internal_hidden_dim, att_dim)
        )
    
    def compute_loss(self, out, batch, free_n=0):
        # Estrai dati...
        mask = batch["mask"]
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
   
        l_pred_bt = loss_prediction(obs_hat, obs, 
                                    rewards_hat, rewards, 
                                    cont_hat,cont, 
                                    user_att_hat, user_att, 
                                    mask=mask, 
                                    beta=self.hparams.beta_pred
        )
        l_dyn_bt  = loss_dynamic(q_logits, p_logits, mask=mask, beta=self.hparams.beta_dyn)
        l_rep_bt  = loss_representation(q_logits, p_logits, mask=mask, beta=self.hparams.beta_rep)
        
        l_kl = l_dyn_bt + l_rep_bt
        
        if free_n > 0:
            l_kl = torch.maximum(l_kl, torch.tensor(free_n, device=l_kl.device))
        
            if mask is not None:
                l_kl = l_kl * mask

        
        loss_total_bt = l_pred_bt + l_kl
        
      
        valid_steps = mask.sum(dim=1).clamp(min=1.0) # [B]
        loss_batch = loss_total_bt.sum(dim=1) / valid_steps # [B]
        
        
        components = {
            "KL_divergence"  : l_kl.sum().item() / mask.sum().item(),
            "prediction_loss": l_pred_bt.sum().item() / mask.sum().item(), 
            "dynamic_loss":    l_dyn_bt.sum().item()  / mask.sum().item(),
            "representation_loss": l_rep_bt.sum().item() / mask.sum().item(),
        }
        
        return loss_batch, components
    
    def setup(self, stage):
        self.unfreeze()
    
    def freeze_model(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_model(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def configure_optimizers(self):
        
        return LaProp(self.parameters(), self.hparams.lr, eps=1e-6)
    
    def encode(self, obs, user_att):
        enc = self.encoder(obs)
        user_att = symlog(user_att)
        att_user = self.att_encoder(user_att)
        
        # -- fuse obs with user att, use simple cat fusion
        enc = torch.cat([enc, att_user], dim=-1)
        embs_encode_state = self.encoder_proj(enc)
        return embs_encode_state

 
    def predict_reward(self, feat):
        return self.reward_head(feat).squeeze(-1) # pred directly in symlog space
    
    def predict_continue(self, feat):
        return self.cont_head(feat).squeeze(-1) # pred directly in symlog space
    
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

        post_h = torch.stack(post_h, dim=1)            
        post_z = torch.stack(post_z, dim=1)            
        
        post_logits  = torch.stack(post_logits, dim=1)  
        prior_logits = torch.stack(prior_logits, dim=1)

        feat = torch.cat([post_h, post_z], dim=-1)     
        feat = self.feat_proj(feat)

        x_hat    = self.predict_obs(feat)                    
        r_hat    = self.predict_reward(feat)
        att_hat  = self.predict_user_att(feat)
        c_logits = self.predict_continue(feat) 


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
        obs = batch["obs"]
        if obs.dtype == torch.uint8:
            batch["obs"] = obs.float().div_(255.0).detach()
        
        out = self.forward(batch["obs"], batch["action"], batch["user_att"], batch["mask"])
        

        loss_batch, components = self.compute_loss(out, batch, self.hparams.free_nats)
        weights = batch["weight"]
        loss = (loss_batch * weights).mean()
        
        #self.log_dict(components, prog_bar=True, on_step=True, on_epoch=True)
        self.log("validation_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
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
        
        priority_idx = batch["priority_idx"].detach().cpu().numpy()
        obs = batch["obs"]

        if obs.dtype == torch.uint8:
            batch["obs"] = obs.float().div_(255.0).detach()
        
        out = self.forward(batch["obs"], batch["action"], batch["user_att"], batch["mask"])
        
        loss_batch, components = self.compute_loss(out, batch)

        weights = batch["weight"]
        loss = (loss_batch * weights).mean()
        
        self.log_dict(components,    prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        #print(f"--- DEBUG STEP {self.global_step} ---")
        #print(f"Obs Range: min={batch['obs'].min()}, max={batch['obs'].max()}")
        #print(f"Weights Esempio (primi 5): {batch['weight'][:5]}")
        #print(f"Mask Sum: {batch['mask'].sum().item()} / {batch['mask'].numel()}")
        #print(f"Prediction Loss Raw: {components['prediction_loss']}")
        # 4. Return per PER e Ottimizzatore
        return {
            "loss": loss,
            "episods_idx": priority_idx,
            "episods_loss": loss_batch.detach().cpu().numpy()
        }
    
    