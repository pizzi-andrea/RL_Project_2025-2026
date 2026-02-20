

import lightning as L
import numpy as np
import torch

from torch import nn
from timm.optim.laprop import LaProp
from torch.optim import Adam
import torchvision
from ..worldModel.dreamer import Dreamer
from .utils import *
from .utils import critic_loss
from .Actor import Actor
from .Critic import Critic

class DreamerAgent(L.LightningModule):
    """Agent based on Dreamer model"""
    def __init__(self, dreamer_model:Dreamer|str, 
                 bukets:int=255,
                 min_buckets:int=-10,
                 max_buckets:int=10, 
                 action_space_dim:int=16, 
                 entropy_coef:int=0.0003,
                 horizont:int=15, 
                 burn_in:int=3, 
                 unimix:float= 0.01, 
                 ema_decay:float= 0.98, 
                 ema_epsilon:float= 1e-6, 
                 lr=0.001, hidden_dim:int=1024, preprocess_callback=None):
        super().__init__()
        self.save_hyperparameters(ignore=['dreamer_model', 'preprocess_callback'])
        # import dreamer model or weigths
        
        if isinstance(dreamer_model, str):
            self.dreamer_model = Dreamer.load_from_checkpoint(dreamer_model)
        else:
            self.dreamer_model = dreamer_model
        
        self.preprocess_callback = preprocess_callback

        # -- configure hyper-parameters
        self.rssm = self.dreamer_model.rssm
        self.entropy_coef = entropy_coef
        self.action_space_dim = action_space_dim
        self.horizont = horizont
        self.burn_in = burn_in

        # -- define EMA scaler (Mobile mean)
        self.ema = EMAScalar(ema_decay, ema_epsilon)
        self.emb_dim = self.dreamer_model.internal_hidden_dim
        
        
        
        # -- Define Actor-critic Structure
        self.actor = Actor(dim_state=self.emb_dim, num_dist=action_space_dim, unimix=unimix, hidden_dim=1024)
        self.critic = Critic(dim_state=self.emb_dim, num_buckets=bukets, min_buckets=min_buckets, max_buckets=max_buckets, hidden_dim=hidden_dim)

    def setup(self, stage):
        """Disable network gradients (different by evaluation mode!)"""
        self.dreamer_model.freeze_model()
    
   # def reparameterize_categorical(logits, tau=1.0):
   #     """Reparametrization trick for OneHot categorical sampling"""
   #     return F.gumbel_softmax(logits, tau=tau, hard=True)
    
    #def reparameterize_bernoulli(logits):
        """Reparametrization trick for Bernoullie sampling"""
    #    probs = torch.sigmoid(logits)
    #    sample = torch.bernoulli(probs)
    #    return probs + (sample - probs).detach()
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val = None, gradient_clip_algorithm = None):
        AGC(self.parameters(), clip_factor=0.03)

    
    def configure_optimizers(self):
        # def parameters
        #   lr: 0.00003
        #   eps: 1e-20
        
        return LaProp(self.parameters(), self.hparams.lr, eps=1e-20)
    
    def training_step(self, batch, batch_idx):
        """Dreamer traning procedure (burn in + dreamer)"""
        
      

        if self.preprocess_callback is not None:
            batch["obs"] = self.preprocess_callback(batch["obs"])
        self.ema = self.ema.train()

        # -- unpack data
        is_first = batch["is_first"] # [B]
        
        
        if batch["obs"].shape[1] > self.burn_in:
            obs = batch["obs"][:, :self.burn_in]
            actions = batch["action"][:, :self.burn_in]
            user_att = batch["user_att"][:, :self.burn_in]
            mask = batch["mask"][:, :self.burn_in]
            # target from replay
            r_buffer = batch["reward"][:, :self.burn_in]
            c_buffer = batch["continue"][:, :self.burn_in]
        else:
           
            obs = batch["obs"]
            actions = batch["action"]
            user_att = batch["user_att"]
            mask = batch["mask"]
             # target from replay
            r_buffer = batch["reward"]
            c_buffer = batch["continue"]
            
        mask_a = batch["mask_action"]

        

        # Passiamo is_first al forward per il reset
        output = self.forward(obs, actions, user_att, is_first, mask, mask_a)

       

        # -- unpack output
        states_hat = output["imagination"] # imm. step
        states_buffer = output["prediction"] # obs step

        r_hat= output["rewards"] # in twohot
        c_hat = output["continues"]
        #a_hat = output["actions"]
        entropy = output["entropy"]
        a_hat_log_probs = output["log_probs"]
        
        

        # -- critic: estimate expected values and lambda-return
        v_hat_log, v_hat = self.critic(states_hat)
        v_hat_log_b, v_hat_b = self.critic(states_buffer)
        
        discount = torch.cumprod(torch.cat([torch.ones_like(c_hat[:, :1]), 0.997 * c_hat[:, :-1]],dim=1),dim=1)
        discount_buffer = torch.cumprod(torch.cat([torch.ones_like(c_buffer[:, :1]), 0.997 * c_buffer[:, :-1]],dim=1),dim=1)
        
          
        with torch.no_grad():
            lambda_return = lambda_returns(
            rewards=r_hat[:, :-1],
            values=v_hat,
            continues=c_hat
            )
            v_bootstrap_replay = lambda_return[:, 0]
            lambda_return_r = lambda_returns(
                rewards=r_buffer[:, :-1],
                values=torch.cat([v_hat_b[:, :-1], v_bootstrap_replay.unsqueeze(1)], dim=1),
                continues=c_buffer
            )

            sym_lambda_return  = symlog(lambda_return)
            sym_lambda_return_b = symlog(lambda_return_r)
            two_hot_lambda_return = twohot(sym_lambda_return, self.critic.buckets)
            two_hot_lambda_return_b = twohot(sym_lambda_return_b, self.critic.buckets)
            #print("r_hat shape:", r_hat.shape)
            #print("r_hat min:", r_hat.min())
            #print("r_hat max:", r_hat.max())
            #print("hat_mean", r_hat.mean())
            
            #print("c_t_tot", (c_hat > 0.5).sum())
            
        
        
        # -- evaluate critic respect lambda return in twohot encoding
        critic_l = critic_loss(
            v_hat_log[:, :-1], 
            two_hot_lambda_return, 
            continues=discount[:, :-1]
        )
        # --  lambda return computed on real values (burn-in data)
        critic_l_reply = critic_loss(
            v_hat_log_b[:, :-1],
            two_hot_lambda_return_b,
            continues=discount_buffer[:, :-1]
        )

        critic_loss_v = critic_l + (0.3 * critic_l_reply)
        
        # -- compute actor loss respect lambda_return and v_hat not in twohot encoding
        actor_loss_v, ema_scale, advantage, _ = actor_loss(
            lambda_return, 
            v_hat[:, :-1], 
            a_hat_log_probs[:, :-1], 
            entropy[:, :-1], 
            self.ema, 
            entropy_scale=self.entropy_coef,
            continues=discount[:, :-1]
        )
        
        # -- ignore terminated steps

        loss = critic_loss_v + actor_loss_v

        valid_mask = discount[:, :-1]
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        with torch.no_grad():
            # Lambda return medio per traiettoria, poi media del batch
            log_lambda = (lambda_return * valid_mask).sum(dim=1) / lengths
            
            # Reward media (attenzione: r_hat qui deve essere il valore scalare, non two-hot)
            # Se r_hat è scalare (es. dopo symexp o decoding):
            log_reward = (r_hat[:, :-1] * valid_mask).sum(dim=1) / lengths
            
            # Entropia media
            log_entropy = entropy
        
        self.log_dict({
            "train_loss"    : loss.item(),
            "actor_loss"    : actor_loss_v.item(),
            "critic_loss"   : critic_loss_v.item(),
            "critic_reply_l"  : critic_l_reply.item(),
            "lambda_return" : log_lambda.mean().item(),
            "mean_reward"   : log_reward.mean().item(),
            "entropy"       :log_entropy.mean().item(),
            "ema_scale"     : ema_scale,
            "advantage"     : advantage
        }, prog_bar=True, on_step=True, on_epoch=False)

        return {
            
            "loss": loss,
            "imagination": output["imagination"][0]
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        obs = outputs["imagination"]
        obs = self.dreamer_model.predict_obs(obs)
        obs = torch.sigmoid(obs)

        if batch_idx == 0:
            n = min(obs.size(0), 5)
            comparison = obs
            grid = torchvision.utils.make_grid(comparison, nrow=n)
            self.logger.experiment.add_image(
                "val_reconstruction", grid, self.current_epoch
            )
        
    def act(self, obs, att_user, old_act, rssm_state, mask_actions=None):
        """Single step agent action"""

        embed_input = self.dreamer_model.encode(obs, att_user)
        
        new_rssm_state, _ = self.rssm.observe_step(rssm_state, old_act, embed_input)

        feature = torch.cat([new_rssm_state.h, new_rssm_state.z], dim=-1)
        feature = self.dreamer_model.feat_proj(feature)
        
        action, _, _ = self.actor(feature, mask_actions)


        action_numpy = action.squeeze(0).cpu().detach().numpy().astype(np.uint8)
        return action_numpy, new_rssm_state
    
    def validation_step(self, batch, batch_idx):
        pass

        
        # --
        #with torch.no_grad():
            
            #decoded_logits = self.dreamer_model.predict_obs(self.trajectory_eval)
            #video_path = Path(self.logger.log_dir).joinpath(f"train_dream_epoch_{self.current_epoch}.avi")
            #make_video(decoded_logits, video_path, fps=30)
        
    def forward(self, obs, action, user_att,is_first, mask=None, mask_actions=None, rssm_state=None) :
        
        B, _, _, _, _ = obs.shape
        # -- encode attribute player with obs
        mask_first = is_first.unsqueeze(-1).float().to(self.device)
        # -- fuse obs with user att, use simple cat fusion
        enc = self.dreamer_model.encode(obs, user_att)

        # -- store temporal informations
        if rssm_state is None:
            rssm_state = self.rssm.init_state(B, obs.device)

        
        reward        = torch.zeros(B, self.horizont, device=obs.device)
        continues     = torch.zeros(B, self.horizont, device=obs.device)
        entropys      = torch.zeros(B, self.horizont, device=obs.device)
        actions_out   = torch.zeros(B, self.horizont, action.shape[-1], device=obs.device)
        log_probs     = torch.zeros(B, self.horizont, device=obs.device)
        latent_states = torch.zeros(B, self.horizont, self.emb_dim, device=obs.device)

        #reward_real        = torch.zeros(B, burn_in, device=obs.device)
        #continues_real     = torch.zeros(B, burn_in, device=obs.device)
        #entropys_real      = torch.zeros(B, burn_in, device=obs.device)
        #actions_out_real   = torch.zeros(B, burn_in, action.shape[-1], device=obs.device)
        #log_probs_real     = torch.zeros(B, burn_in, device=obs.device)
        latent_states_real = torch.zeros(B, self.burn_in, self.emb_dim, device=obs.device)
        
        # -- burn-in rssm initialization
        burn_in = self.burn_in if self.training else 0

        with torch.no_grad():
            for t in range(burn_in):
                #rssm_state.h = rssm_state.h * (1.0 - mask_first)
                #rssm_state.z = rssm_state.z * (1.0 - mask_first)
                embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
                embed = self.dreamer_model.feat_proj(embed)
                enc_t = enc[:, t]
                action_t = action[:, t]
                m = mask[:, t]
                latent_states_real[:, t] = embed
                #continues_real[:, t] = self.dreamer_model.predict_continue(embed)
                rssm_state, _ = self.rssm.observe_step(rssm_state, action_t, enc_t, m)
        
        # -- for security detach gradients
        # coerent reset for imagination steps
        rssm_state.h = rssm_state.h * (1.0 - mask_first)
        rssm_state.z = rssm_state.z * (1.0 - mask_first)
        
        # stop gradients for immagination phase
        rssm_state.z = rssm_state.z.detach()
        rssm_state.h = rssm_state.h.detach()
        rssm_state.z_logits = rssm_state.z_logits.detach()

        # -- immagination using rssm
        for t in range(self.horizont):
            
            # define embedding from rssm state (detterministic + probab)
            embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
            embed = self.dreamer_model.feat_proj(embed)
            
            latent_states[:, t] = embed #.detach()

            # get reward from dreamer model
            _, r_t = self.dreamer_model.predict_reward(embed)
            
            # get continue flag from model
            c_t = self.dreamer_model.predict_continue(embed)
            
            continues[:, t] = c_t
            reward[:, t] = r_t
            
            # use actor to predict next action
            act_t, dist_a, entropy = self.actor(embed, mask_actions)

            entropys[:, t] = entropy
            actions_out[:, t] = act_t
            log_probs[:, t] = dist_a.log_prob(act_t)

            rssm_state, _ = self.rssm.imagine_step(rssm_state, act_t)
            #rssm_state.z = rssm_state.z.detach()
            #rssm_state.h = rssm_state.h.detach()
            #rssm_state.z_logits = rssm_state.z_logits.detach()
        
        # -- pack values
        return {
            "rewards"       : reward, 
            "continues"     : continues,
            "entropy"       : entropys, 
            "imagination"   : latent_states,
            "actions"       : actions_out,
            "log_probs"     : log_probs,
            #"continue_r"    : continues_real,
            "prediction"    : latent_states_real 
        }

        
            


                
        
        
