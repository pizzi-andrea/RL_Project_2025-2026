import torchvision
from ..encoder.encoders import Encoder 
from ..encoder.decoders import Decoder
from typing import Dict
from timm.optim.laprop import LaProp
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Optional

# Vision Encoders
# RNN network
from ..dreamer.rssm import RSSM, RSSMState
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
        decode_shape: tuple[int,int,int] = (3, 64,64),
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
        alpha_continue:float=1.0,
        alpha_reward:float=1.0,
        
        unimix: float = 0.01,
       
        encoder = None,
        decoder: Optional[str]|Decoder|None = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.min_buckets = min_buckets
        self.max_buckets = max_buckets
        self.free_nats = free_nats
        self.alpha_obs = alpha_obs
        self.alpha_att  = alpha_att
        self.alpha_continue = alpha_continue
        self.alpha_reward = alpha_reward

        # TODO: rimuovere se necessario
       
        self.rssm = RSSM(
            action_dim=internal_hidden_dim,
            h_dim=h_dim,
            obs_dim=rssm_hidden_dim,
            stoch_latents=stoch_latents,
            stoch_classes=stoch_classes,
            att_dim= internal_hidden_dim,
            hidden_dim=rssm_hidden_dim,
            unimix=unimix,
        )
        
        self.encoding_dim = self.rssm.L * self.rssm.K 

        self.state_dim = self.rssm.hidden_dim  # h_dim + L*K
        self.internal_hidden_dim  = internal_hidden_dim
        self.burn_in = burn_in
        self.unimix = unimix

        if isinstance(Encoder, str):
            self.encoder = Encoder.load_from_checkpoint(encoder, map_location='cpu')
        elif isinstance(encoder, Encoder):
            self.encoder = encoder
        elif not encoder:
            self.encoder = Encoder(model_dim=self.internal_hidden_dim, latent_dim=internal_hidden_dim, rssm_state_dim=self.rssm.hidden_dim)
        else:
            assert "No encoder instance"

        
        if isinstance(decoder, str):
            self.decoder = Decoder.load_from_checkpoint(encoder, map_location='cpu')
        elif isinstance(decoder, Decoder):
            self.decoder = decoder
        elif not decoder:
            self.decoder = Decoder(model_dim=self.internal_hidden_dim, world_model_dim=self.rssm.hidden_dim, input_res=self.encoder.input_res)
        else:
            assert "No encoder instance"

        
        # -- encoders adapter
        self.encoder = EncoderAdapter(self.encoder)

        # -- decoder adapter
        self.decoder = DecoderAdapter(self.decoder)

        self.att_encoder = nn.Sequential(
            nn.Linear(att_dim, internal_hidden_dim*2),
            nn.RMSNorm(internal_hidden_dim*2),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim*2, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),

            
        )

        self.action_encoder = nn.Sequential(

            nn.Linear(action_dim, internal_hidden_dim*2),
            nn.RMSNorm(internal_hidden_dim*2),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim*2, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
        )


        # -- controllers

        # reward predictor



        
        self.register_buffer("buckets", torch.linspace(min_buckets, max_buckets, buckets))
        
        # reward network ############################################
        self.reward_head = nn.Sequential(
            nn.Linear(self.state_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
            
            nn.Linear(internal_hidden_dim, buckets),
    
        )
        ##############################################################

        self.softmax = nn.Softmax(dim=-1)

        # continue predictor
        self.cont_head = nn.Sequential(
            nn.Linear(self.state_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim, 1),
        )

        # attribute predictor
        self.att_head = nn.Sequential(
            nn.Linear(self.state_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
            
            nn.Linear(internal_hidden_dim, att_dim)
        )

        self.state_encoder = nn.Sequential(
           
            nn.Linear(self.rssm.h_dim + self.rssm.L * self.rssm.K, internal_hidden_dim*2),
            nn.RMSNorm(internal_hidden_dim*2),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim*2, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),

            nn.Linear(internal_hidden_dim, internal_hidden_dim),
            nn.RMSNorm(internal_hidden_dim),
            nn.SiLU(),
        )

        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)
        nn.init.zeros_(self.cont_head[-1].bias)
    
    def compute_loss(self, out, batch):

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
        cont_hat = out["c"]
        user_att_hat = out["att_hat"]
        buckets_log = out["buckets_log"]

       

        
        with torch.no_grad():
            user_att = symlog(user_att) # symlog target user attribute
            #obs = (obs / 255)             # imgs normalization
            
            two_hot_rewards = twohot(symlog(rewards), self.buckets)  # robust encoding for high variance prediction (rewards model)
        ######################################################################################

        # recostruction world attribute loss
        l_pred_bt, components = loss_prediction(
            obs_hat, obs, 
            buckets_log, two_hot_rewards, 
            cont_hat,     cont,
            user_att_hat, user_att, 
            mask=mask,
            beta_x=self.alpha_obs, beta_att=self.alpha_att, beta_c=self.alpha_continue, beta_r=self.alpha_reward
        )
        
        #######################################################################################

        # dynamic loss ########################################################################
        l_dyn_bt  = loss_dynamic(q_logits, p_logits, unimix=self.unimix, mask=mask)
        l_rep_bt  = loss_representation(q_logits, p_logits, mask=mask, unimix=self.unimix)
        #######################################################################################
        
        if self.free_nats > 0:
            l_dyn_bt = torch.maximum(l_dyn_bt, torch.tensor(self.free_nats, device=l_dyn_bt.device))
            l_rep_bt = torch.maximum(l_rep_bt, torch.tensor(self.free_nats, device=l_rep_bt.device))
       
        # balance different loss ################################################################
        l_dyn_bt    = l_dyn_bt * self.hparams.beta_dyn
        l_rep_bt    = l_rep_bt * self.hparams.beta_rep
        l_pred_bt   = l_pred_bt * self.hparams.beta_pred
        ########################################################################################
        
        loss_t = l_pred_bt + l_dyn_bt + l_rep_bt # [B,T]
        
      
        length     = mask.sum(dim=1).clamp(min=1).detach()
        loss_batch = loss_t.sum(-1) / length
        
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
    
    def encode(self, obs, user_att): # x_t encoding
        """Encode observation + world attributes"""
        return self.encode_obs(obs), self.encode_att(user_att)
        # -- fuse obs with user att, use simple cat fusion
    
    def encode_obs(self, x):
        x = (x / 255)
        x = self.encoder(x)
        return x 
    
    def encode_att(self, x):
        x = symlog(x) 
        x = self.att_encoder(x)

        return x

    def encode_action(self, x):
        return self.action_encoder(x)
    
    def encode_state(self, h, z):
        x = torch.cat([h, z], dim=-1)
        return self.state_encoder(x)

 
    def predict_reward(self, feat):
        """Predict reward in differentiable world"""
        # 1. Calcola i logit (pesi non normalizzati) per ogni bucket
        pred_rewards_buckets_log = self.reward_head(feat)
        
        # 2. Converti i logit in probabilità tramite softmax
        pred_rewards_buckets_probs = self.softmax(pred_rewards_buckets_log)

        # 3. Calcola l'aspettativa della reward nello spazio symlog.
        # Effettuiamo il prodotto scalare (somma ponderata) tra le probabilità
        # e i valori target (self.buckets) lungo l'ultima dimensione[cite: 7].
        expected_rewards_symlog = torch.sum(pred_rewards_buckets_probs * self.buckets, dim=-1)
        
        # 4. Ritorna nello spazio lineare tramite symexp
        pred_rewards = symexp(expected_rewards_symlog)
        
        # Restituisci sempre sia i logit che la reward predetta, come richiesto
        return pred_rewards_buckets_log, pred_rewards
    
  

    def predict_continue(self, feat):
        """
        predict continue flags, return both values logits and value to obtain maximum stability in the training
        """
        logits = self.cont_head(feat).squeeze(-1)
        probs = torch.sigmoid(logits) 
        
        
        return logits, probs
    
    def predict_obs(self, s):
        """Recostruct observed state"""
        
        obs = self.decoder(s)
        
        return obs 
    
    def predict_user_att(self, feat):
        """Predict attribute of differentiable world"""
        return self.att_head(feat)

    def _get_loss_for_batch(self, episodes_idx, loss_episodes):
        """get loss for batch and index episodes in batch"""
        idx_episodes = episodes_idx.detach().cpu().numpy()
        loss_episodes = loss_episodes.detach().cpu().numpy()
        
        return {
            "episodes_idx":idx_episodes, 
            "episodes_loss":loss_episodes
        }
    
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val = None, gradient_clip_algorithm = None):
        AGC(self.parameters(), clip_factor=0.3)
        
    
        
            
    def training_step(self, batch, batch_idx):
        priority_idx = batch["priority_idx"].detach().cpu().numpy()
        batch["obs"] = batch["obs"].float()
        is_first = batch["is_first"] 

        if self.burn_in > 0:
            # 1. Burn Data: prendiamo sempre i primi 'burn_in' frame per il warm-up
            burn_data = {
                "obs":      batch["obs"][:, :self.burn_in],
                "action":   batch["action"][:, :self.burn_in],
                "user_att": batch["user_att"][:, :self.burn_in],
                "mask":     batch["mask"][:, :self.burn_in]
            }
            
            T_total = batch["obs"].shape[1]
            T_train = T_total - self.burn_in
            
            # 2. Train Data con select_data:
            # Se is_first=True -> train_data = batch[:, :T_train] (inizia da 0)
            # Se is_first=False -> train_data = batch[:, burn_in:] (inizia dopo il burn-in)
            train_data = {
                "obs":      select_data(batch["obs"], is_first, self.burn_in, T_train),
                "action":   select_data(batch["action"], is_first, self.burn_in, T_train),
                "user_att": select_data(batch["user_att"], is_first, self.burn_in, T_train),
                "mask":     select_data(batch["mask"], is_first, self.burn_in, T_train),
                "continue": select_data(batch["continue"], is_first, self.burn_in, T_train),
                "reward":   select_data(batch["reward"], is_first, self.burn_in, T_train),
                "is_first": is_first, 
                "weight":   batch["weight"]
            }
        else:
            train_data = batch
            burn_data = None

        out = self.forward(train_data, burn_data)
        
        # Calcolo Loss (su train_data che ora include gli step iniziali se is_first)
        loss_batch, components = self.compute_loss(out, train_data)

        weights = train_data["weight"]
        loss = (loss_batch * weights).mean()
        
        # Logging...
        self.log_dict(components, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return {
            "loss": loss,
            "episodes_idx": priority_idx,
            "episodes_loss": loss_batch.detach().cpu().numpy(),
            "x_hat": out["x_hat"][0].detach().cpu(),
            "x": train_data["obs"][0].detach().cpu() # Logghiamo obs di train_data
        }
    
    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        
        x_hat = torch.sigmoid(outputs["x_hat"]) # [B, C, H, W]
        x = (outputs["x"] /255)
        
        
        if batch_idx == 0: # use only first batch for logging
            x_log = torch.cat([x[:8], x_hat[:8]], dim=0)
            grid = torchvision.utils.make_grid(x_log[:16], nrow=8)
            self.logger.experiment.add_image("reconstruction", grid, self.global_step)
    


    def on_train_start(self):
        self.unfreeze()
        self.rssm.unfreeze()
    
    def forward(self, train_data, burn_in_data=None) -> Dict[str, torch.Tensor]:
        obs      = train_data["obs"]
        actions  = train_data["action"]
        att_user = train_data["user_att"]
        mask     = train_data["mask"]
        is_first = train_data["is_first"]   # [B] Tensor booleano

        B, T   = obs.shape[0], obs.shape[1]
        device = obs.device

        state       = self.rssm.init_state(B, device=device)
        last_action = torch.zeros_like(actions[:, 0])
        
        h_dim = self.rssm.h_dim
        L, K  = self.rssm.L, self.rssm.K

        # --- 1. Fase di Burn-in Dinamico (Rispetta il Padding) ---
        if burn_in_data is not None:
            with torch.no_grad():
                for t in range(burn_in_data["obs"].shape[1]):
                    # Maschera al tempo t: 1 = informazione, 0 = padding
                    m_t = burn_in_data["mask"][:, t].unsqueeze(-1) # [B, 1]
                    
                    x_t, att_t = self.encode(burn_in_data["obs"][:, t], burn_in_data["user_att"][:, t])
                    a_t = self.encode_action(last_action)
                    
                    # Calcolo del prossimo stato potenziale
                    state, _, _ = self.rssm.observe_step(
                        state, 
                        a_t, 
                        x_t, 
                        att_t, 
                        burn_in_data["mask"][:, t]
                    )
                    
                    
                    # Aggiorniamo l'azione solo se il frame corrente era valido
                    last_action = torch.where(m_t.bool(), burn_in_data["action"][:, t], last_action)

        # --- 2. Risoluzione Conflitto is_first ---
        # Se is_first è True, la sequenza di training è l'inizio di un episodio (t=0).
        # Resettiamo lo stato a 'fresh' ignorando qualsiasi riscaldamento precedente.
        state, last_action = self._apply_episode_reset(state, last_action, is_first)
       
        # --- 3. Fase di Training (Loop Principale) ---
        post_h       = torch.zeros(B, T, h_dim, device=device)
        post_logits  = torch.zeros(B, T, L, K,  device=device)
        prior_logits = torch.zeros(B, T, L, K,  device=device)
        post_z       = torch.zeros(B, T, L*K , device=device)

        for t in range(T):
            x_t, att_t = self.encode(obs[:, t], att_user[:, t])
            a_t = self.encode_action(last_action)
            state, prior_logits_t, post_logits_t = self.rssm.observe_step(state, a_t, x_t, att_t, mask[:, t])
            
            post_h[:,  t]      = state.h
            post_z[:,  t]      = state.z.flatten(start_dim=1)
            post_logits[:,  t] = post_logits_t   
            prior_logits[:, t] = prior_logits_t  

            last_action = actions[:, t]

        # --- 4. Predizioni Finali ---
        s_t = self.encode_state(post_h, post_z) 
        
        x_hat        = self.predict_obs(s_t)             
        r_hat_log, _ = self.predict_reward(s_t)             
        att_hat      = self.predict_user_att(s_t)                
        c, _            = self.predict_continue(s_t) 

        return {
            "x_hat": x_hat,
            "c": c,
            "post_logits": post_logits,
            "prior_logits": prior_logits,
            "att_hat": att_hat,
            "buckets_log": r_hat_log
        }


    def _apply_episode_reset(
        self,
        state: RSSMState,
        last_action: torch.Tensor,
        is_first: torch.Tensor,        # [B]
    ) -> tuple[RSSMState, torch.Tensor]:
        """
        Episodio nuovo → stato fresco da init_state, non moltiplicazione per maschera.
        Rende l'intento esplicito e non si confonde con il padding dell'RSSM.
        """
        if is_first is None or not is_first.any():
            return state, last_action

        # Stato pulito di riferimento
        fresh = self.rssm.init_state(state.h.shape[0], device=state.h.device)

        # is_first=1 → prendi da fresh, is_first=0 → tieni state
        # stop-gradient: is_first non partecipa al grafo computazionale
        with torch.no_grad():
            first  = is_first.float().unsqueeze(-1)          # [B, 1]
            first3 = first.unsqueeze(-1)                     # [B, 1, 1]

        state.h        = torch.where(first.bool(),  fresh.h,        state.h)
        state.z = torch.where(first3.bool(), fresh.z, state.z)
        last_action    = torch.where(first.bool(),
                                    torch.zeros_like(last_action),
                                    last_action)

        return state, last_action
     
        
    
    