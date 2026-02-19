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
        burn_in:int= 10,
        buckets:int = 255,
        min_buckets:int=-20,
        max_buckets:int=20,
        lr: float = 3e-4,
        beta_pred: float = 1.0,
        beta_dyn: float = 1.0,
        beta_rep: float = 0.1,
        free_nats: float = 1.0,
        alpha_obs:float=1.0,
        alpha_att:float=1.0,
        aplha_continue:float=1.0,
        alpha_reward:float=1.0,
        
        unimix: float = 0.01,
        # --- nuove opzioni VAE --- 
        vae: Optional[str]|VAE|None = None,
        preprocess_callback=None
        
        # se vuoi che enc_x sia esattamente latent_dim del VAE, imposta enc_dim=vae_latent_dim
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae', 'preprocess_callback'])
        self.min_buckets = min_buckets
        self.max_buckets = max_buckets

        self.alpha_obs = alpha_obs
        self.alpha_att  = alpha_att
        self.aplha_continue = aplha_continue
        self.alpha_reward = alpha_reward
        
        self.rssm = RSSM(
            action_dim=action_dim,
            deter_dim=h_dim,
            obs_dim=internal_hidden_dim,
            stoch_latents=stoch_latents,
            stoch_classes=stoch_classes,
            embed_dim=rssm_hidden_dim,
            unimix=unimix,
        )

        self.preprocess_callback = preprocess_callback
        rssm_feat_dim = h_dim + (stoch_latents * stoch_classes) # h_dim + L*K
        self.internal_hidden_dim  = internal_hidden_dim
        self.burn_in = burn_in

        if isinstance(vae, str):
            self.vae = VAE.load_from_checkpoint(vae, map_location='cpu')
        elif isinstance(vae, VAE):
            self.vae = vae
        elif not vae:
            self.vae = VAE(latent_dim=internal_hidden_dim)
        else:
            assert "No VAE instance"

        # -- VAE encoders adapter
        self.encoder = VAEEncoderAdapter(self.vae, out_dim=internal_hidden_dim)

        # -- VAE decoder adapter
        self.decoder = VAEDecoderAdapter(self.vae, feat_dim=internal_hidden_dim, decode_shape=decode_shape)

        self.att_encoder = nn.Sequential(
            nn.Linear(att_dim, out_features=internal_hidden_dim),
            #nn.RMSNorm(internal_hidden_dim),
            #nn.SiLU()
        )

        self.encoder_proj = nn.Sequential(
            nn.Linear(2*internal_hidden_dim, internal_hidden_dim),
            #nn.RMSNorm(internal_hidden_dim),
            #nn.SiLU()
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(rssm_feat_dim, internal_hidden_dim),
            #nn.RMSNorm(internal_hidden_dim),
            #nn.SiLU()
        )


        # -- controllers

        # reward predictor

        self.register_buffer("buckets", torch.linspace(min_buckets, max_buckets, buckets) ) 
        self.reward_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim*2),
            nn.RMSNorm(internal_hidden_dim*2),
            nn.SiLU(),
            
            nn.Linear(internal_hidden_dim*2, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
            
            nn.Linear(internal_hidden_dim, buckets),
    
        )
        self.softmax = nn.Softmax(dim=-1)

        # continue predictor
        self.cont_head = nn.Sequential(
            nn.Linear(internal_hidden_dim, internal_hidden_dim*2),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim*2, internal_hidden_dim),
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
        # unpack data
        mask = batch["mask"]
        obs = batch["obs"]           
        rewards = batch["reward"]    
        cont = batch["continue"]         
        user_att = batch["user_att"] 
        mask = batch["mask"]         
        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]

        # unpack outputs
        obs_hat = out["x_hat"]
        #rewards_hat = out["r_hat"]
        cont_hat = out["c_logits"]
        user_att_hat = out["att_hat"]
        buckets_log = out["buckets_log"]

        # scale trasformations for normalize values across different scenarios, encode targets
        with torch.no_grad():
            user_att = symlog(user_att)
            rewards  = symlog(rewards)
            two_hot_rewards = twohot(rewards, self.buckets)
        ######################################################################################

        # recostruction world attribute loss
        l_pred_bt, components = loss_prediction(
            obs_hat, obs, 
            buckets_log, two_hot_rewards, 
            cont_hat,cont,
            user_att_hat, user_att, 
            mask=mask, 

            beta_x=self.alpha_obs,
            beta_att=self.alpha_att,
            beta_c=self.aplha_continue,
            beta_r=self.alpha_reward
        )
        #######################################################################################

        # dynamic loss ########################################################################
        l_dyn_bt  = loss_dynamic(q_logits, p_logits, mask=mask)
        l_rep_bt  = loss_representation(q_logits, p_logits, mask=mask)
        #######################################################################################
        
        
        # free nats for dynamic ###############################################################
        if free_n > 0:
            l_dyn_bt = torch.maximum(l_dyn_bt, torch.tensor(free_n, device=l_dyn_bt.device))
            l_rep_bt = torch.maximum(l_rep_bt, torch.tensor(free_n, device=l_dyn_bt.device))
        #######################################################################################

        # balance different loss ################################################################
        l_dyn_bt    = l_dyn_bt * self.hparams.beta_dyn
        l_rep_bt    = l_rep_bt * self.hparams.beta_rep
        l_pred_bt   = l_pred_bt * self.hparams.beta_pred
        ########################################################################################
        
        loss_t = l_pred_bt + l_dyn_bt + l_rep_bt # [B,T]
        
      
        length = mask.sum(dim=1).clamp(min=1).detach() # [t]

        # weighted mean for batch [B]
        loss_batch = loss_t.sum(-1) / length # [B]
        
        # simple loss, long and short sequence same dominance (scalar)
        
        length = length.cpu()
        
        
        
        components |= {
            "kl_divergence"         :      ( (l_dyn_bt+l_rep_bt).cpu().sum(-1) / length).mean(),
            "prediction_loss"       :      (l_pred_bt.detach().cpu().sum(-1)   / length).mean(), 
            "dynamic_loss"          :         (l_dyn_bt.detach().cpu().sum(-1) / length).mean(),
            "representation_loss"   :  (l_rep_bt.detach().cpu().sum(-1)         / length).mean(),
        }
        
        return loss_batch, components
    
    def setup(self, stage):
        self.unfreeze()
    
    def freeze_model(self):
        """Disable network gradients"""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_model(self):
        """Enable network gradients"""
        for p in self.parameters():
            p.requires_grad = True
    
    def configure_optimizers(self):
        return LaProp(self.parameters(), self.hparams.lr, eps=1e-20)
    
    def encode(self, obs, user_att):
        """Encode observation + world attributes"""
        enc = self.encoder(obs)
        user_att = symlog(user_att) # transform in symlog space
        att_user = self.att_encoder(user_att)
        
        # -- fuse obs with user att, use simple cat fusion
        enc = torch.cat([enc, att_user], dim=-1)
        embs_encode_state = self.encoder_proj(enc)
        return embs_encode_state

 
    def predict_reward(self, feat):
        """Predict reward in differentiable world"""
        B = self.buckets
        pred_rewards_buckets_log = self.reward_head(feat)
        pred_rewards_buckets_probs = self.softmax(pred_rewards_buckets_log)

        pred_symlog = (pred_rewards_buckets_probs * B).sum(dim=-1)
        pred_rewards = symexp(pred_symlog, self.min_buckets, self.max_buckets)
        return pred_rewards_buckets_log, pred_rewards.squeeze(-1)
    
    # In network/worldModel/dreamer.py

    def predict_continue(self, feat):
        """
        Restituisce la probabilità di continuazione (Soft) per il calcolo della Loss.
        """
        logits = self.cont_head(feat).squeeze(-1)
        probs = torch.sigmoid(logits) 
        
        # Per DreamerV3 standard (Loss Weighting), restituiamo la probabilità.
        # Non usiamo STE qui perché vogliamo che la Loss sia pesata in modo differenziabile e soft.
        return probs
    
    def predict_obs(self, feat):
        """Recostruct observed state"""
        return self.decoder(feat) 
    
    def predict_user_att(self, feat):
        """Predict attribute of differentiable world"""
        return self.att_head(feat)

    def _get_loss_for_batch(self, episods_idx, loss_episods):
        """get loss for batch and index episods in batch"""
        idx_episods = episods_idx.detach().cpu().numpy()
        loss_episods = loss_episods.detach().cpu().numpy()
        
        return {
            "episods_idx":idx_episods, 
            "episods_loss":loss_episods
        }
    
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val = None, gradient_clip_algorithm = None):
        AGC(self.parameters(), clip_factor=0.03)
        
    
        
            
    def training_step(self, batch, batch_idx):
        
        # indexs need to update PER buffer not require tensor
        priority_idx = batch["priority_idx"].detach().cpu().numpy()

        # masks not required grads
        batch["mask_action"] =  batch["mask_action"].detach()
        batch["mask"] = batch["mask"].detach()
    
        # normalize observation in [0,1] range
        if self.preprocess_callback is not None:
            batch["obs"] = self.preprocess_callback(batch["obs"])
       
        is_first = batch["is_first"] # [B] Tensor bool
        
        if batch["obs"].shape[1] > self.burn_in:
            # 1. Burn Data: prendiamo sempre i primi 'burn_in' frame
            burn_data = {
                "obs": batch["obs"][:, :self.burn_in],
                "action": batch["action"][:, :self.burn_in],
                "user_att": batch["user_att"][:, :self.burn_in],
                "mask": batch["mask"][:, :self.burn_in]
            }
            
            
            
            T_total = batch["obs"].shape[1]
            T_train = T_total - self.burn_in
            
            

            train_data = {
                "obs":      select_data(batch["obs"], is_first, self.burn_in, T_train),
                "action":   select_data(batch["action"], is_first, self.burn_in, T_train),
                "user_att": select_data(batch["user_att"], is_first, self.burn_in, T_train),
                "mask":     select_data(batch["mask"], is_first, self.burn_in, T_train),
                "continue": select_data(batch["continue"], is_first, self.burn_in, T_train),
                "reward":   select_data(batch["reward"], is_first, self.burn_in, T_train),
                "is_first": is_first, # Passalo al forward per il reset dello stato
                "weight":   batch["weight"]
            }
        else:
            train_data = batch
            burn_data = None
            

        out = self.forward(train_data, burn_data)
        
        # compute model loss
        loss_batch, components = self.compute_loss(out, train_data, self.hparams.free_nats)

        # add PER weights to remove sampling bias
        weights = train_data["weight"]

       
        loss = (loss_batch * weights).mean() #[1]
        
        # logging
        self.log_dict(components,    prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        #print(f"--- DEBUG STEP {self.global_step} ---")
        #print(f"Obs Range: min={batch['obs'].min()}, max={batch['obs'].max()}")
        #print(f"Weights Esempio (primi 5): {batch['weight'][:5]}")
        #print(f"Mask Sum: {batch['mask'].sum().item()} / {batch['mask'].numel()}")
        #print(f"Prediction Loss Raw: {components['prediction_loss']}")
        # 4. Return per PER e Ottimizzatore

        # ouput train step
        return {
            "loss": loss,
            "episods_idx": priority_idx,
            "episods_loss": loss_batch.detach().cpu().numpy(),
            "x_hat": out["x_hat"][0].detach().cpu(),
            "x": batch["obs"][0].detach().cpu()
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        obs = outputs["x"]
        obs_hat = torch.sigmoid(outputs["x_hat"])

        if batch_idx == 0:
            n = min(obs.size(0), 8)
            comparison = torch.cat([obs[:n], obs_hat[:n]], dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=n)
            self.logger.experiment.add_image(
                "val_reconstruction", grid, self.current_epoch
            )
    


    
    
    def forward(self, train_data, burn_indata=None) -> Dict[str, torch.Tensor]:
        obs = train_data["obs"]
        
        
        B, T, _, _, _ = obs.shape
        is_first = train_data["is_first"]
        # -- init memory
        state = self.rssm.init_state(B, device=obs.device)
        burn_in = self.burn_in if self.training else 0
        
        # -- burn_in precedurre
        
        if burn_indata is not None:
            actions = burn_indata["action"]
            att_user = burn_indata["user_att"]
            mask = burn_indata["mask"]
            obs = burn_indata["obs"]
            with torch.no_grad():
                
                for t in range(burn_in):
                    a_t = actions[:, t]
                    att_t = att_user[:, t]
                    m_t = mask[:, t]
                    obs_t = obs[:, t]
                    
                    # Encoding dell'osservazione
                    x_t = self.encode(obs_t, att_t)
                    
                    # Step RSSM (aggiorna solo lo stato)
                    mask_first = is_first.unsqueeze(-1).float()
                    state.h = state.h * (1.0 - mask_first)
                    state.z = state.z * (1.0 - mask_first)
                    state, _ = self.rssm.observe_step(state, a_t, x_t, m_t)
            
            z = state.z.detach()
            h = state.h.detach()
            z_log = state.z_logits.detach()

            state.z = z
            state.h = h 
            state.z_logits = z_log
        #####################################################################
       
        mask_first = is_first.unsqueeze(-1).float()
        state.h = state.h * (1.0 - mask_first)
        state.z = state.z * (1.0 - mask_first)
       
        # -- store temp data
        actions = train_data["action"]
        att_user = train_data["user_att"]
        mask = train_data["mask"]
        obs = train_data["obs"]


        B, T = obs.shape[0], obs.shape[1]
        h_dim = self.rssm.deter_dim
        z_dim = self.rssm.L * self.rssm.K
        L, K = self.rssm.L, self.rssm.K

        # allocate buffers
        post_h      = torch.zeros(B, T, h_dim, device=obs.device)
        post_z      = torch.zeros(B, T, z_dim, device=obs.device)
        post_logits = torch.zeros(B, T, L, K, device=obs.device)
        prior_logits = torch.zeros(B, T, L, K, device=obs.device)

        # compute real length for data and ingnore padding during iteration
        
        
        # -- compute loss
        for t in range(T):
            a_t = actions[:, t]
            att_t = att_user[:, t]
            m_t = mask[:,t]
            obs_t = obs[:, t]
            #print(f"att shape: {att.shape}")
            x_t = self.encode(obs_t, att_t)  # encode observation + user attributes
            state, plogits = self.rssm.observe_step(state, a_t, x_t, m_t)
            post_h[:, t] = state.h
            post_z[:, t] = state.z
            post_logits[:, t] = state.z_logits
            prior_logits[:, t] = plogits
        ############################################################################
       
       
        # feactures vector
        feat = torch.cat([post_h, post_z], dim=-1)     
        feat = self.feat_proj(feat)

        post_h = None
        post_z = None
        # predict world attributes
        x_hat                   = self.predict_obs(feat)        # latent space              
        r_hat_logits            = self.reward_head(feat)     # linear + twohot
        att_hat                 = self.att_head(feat)   # symlog
        c_logits                = self.cont_head(feat).squeeze(-1)   # linear (logits)


        return {
            "x_hat": x_hat,
            #"r_hat": r_hat, note need for regression task
            "c_logits": c_logits,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
            "att_hat": att_hat,
            "buckets_log": r_hat_logits
        }
     
        
    
    