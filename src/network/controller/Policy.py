

import lightning as L
import numpy as np
import torch

from torch import nn
from timm.optim.laprop import LaProp
from ..worldModel.dreamer import Dreamer
from .utils import *
from .Actor import Actor
from .Critic import Critic

class DreamerAgent(L.LightningModule):
    """Agent based on Dreamer model"""
    def __init__(self, dreamer_model:Dreamer|str, bukets=255, action_space_dim:int=16, entropy_coef:int=0.0003,horizont:int=15, burn_in:int=3, unimix:float= 0.01, ema_decay:float= 0.98, ema_epsilon:float= 1e-6, lr=0.001, hidden_dim:int=1024, one_hot:bool=True):
        super().__init__()
        self.save_hyperparameters(ignore=["dreamer_model"])
        # import dreamer model or weigths
        
        if isinstance(dreamer_model, str):
            self.dreamer_model = Dreamer.load_from_checkpoint(dreamer_model)
        else:
            self.dreamer_model = dreamer_model
        
        

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
        self.critic = Critic(dim_state=self.emb_dim, num_buckets=bukets, hidden_dim=hidden_dim)

    def setup(self, stage):
        """Disable network gradients (different by evaluation mode!)"""
        self.dreamer_model.freeze_model()
    
    def reparameterize_categorical(logits, tau=1.0):
        """Reparametrization trick for OneHot categorical sampling"""
        return F.gumbel_softmax(logits, tau=tau, hard=True)
    
    def reparameterize_bernoulli(logits):
        """Reparametrization trick for Bernoullie sampling"""
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs)
        return probs + (sample - probs).detach()
    

    def configure_optimizers(self):
        # def parameters
        #   lr: 0.00003
        #   eps: 1e-20
        return LaProp(self.parameters(), lr=self.hparams.lr, eps=1e-20)

    def training_step(self, batch, batch_idx):
        """Dreamer traning procedure (burn in + dreamer)"""
        
        # -- unpack data
        obs     = batch["obs"][:, 0:self.burn_in]
        actions = batch["action"][:, 0:self.burn_in]
        user_att = batch["user_att"][:, 0:self.burn_in]
        mask     = batch["mask"][:, 0:self.burn_in]
        mask_a   = batch["mask_action"]

        # normalize and convert standard image in float32 tensor
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0).detach()

        # generate new immaginary trajectory using rssm model
        output = self.forward(obs, actions, user_att, mask, mask_a)

        # -- unpack output
        states_hat = output["imagination"]
        r_hat= output["rewards"] # in twohot
        c_hat = output["continues"]
        #a_hat = output["actions"]
        entropy = output["entropy"]
        a_hat_log_probs = output["log_probs"]
        

        # -- critic: estimate expected values and lambda-return
        v_hat_log, v_hat = self.critic(states_hat)

        # r_hat and v_hat are just in same scales and encoding with twohot method
        
        lambda_return = lambda_returns(
            rewards=r_hat,
            values=v_hat,
            continues=c_hat
        )
        
        # encode target using twohot function for loss
        with torch.no_grad():
            two_hot_lambda_return = twohot(lambda_return, self.critic.buckets)
        
        
        # -- evaluate critic respect lambda return in twohot encoding
        critic_l = critic_loss(
            v_hat_log[:, :-1], 
            two_hot_lambda_return, 
            continues=c_hat[:, :-1]
        )
        
        # -- compute actor loss respect lambda_return and v_hat not in twohot encoding
        actor_l, ema_scale = actor_loss(
            lambda_return, 
            v_hat[:, :-1], 
            a_hat_log_probs[:, :-1], 
            entropy[:, :-1], 
            self.ema, 
            entropy_scale=self.entropy_coef,
            continues=c_hat[:, :-1]
        )
        
        # -- ignore terminated steps

        loss = critic_l + actor_l
        
        self.log_dict({
            "train_loss"    : loss.detach().item(),
            "actor_loss"    : actor_l.detach().item(),
            "critic_loss"   : critic_l.detach().item(),
            "lambda_return" : lambda_return.mean().item(),
            "mean_reward"   : r_hat.detach().mean().item(),
            "entropy"       : entropy.detach().mean().item(),
            "ema_scale"     : ema_scale
        }, prog_bar=True, on_step=True, on_epoch=True)

        return loss
        
    def act(self, obs, att_user, old_act, rssm_state):
        obs_t = obs.unsqueeze(0).unsqueeze(0)
        att_t = att_user.unsqueeze(0).unsqueeze(0)

        enc_embed = self.dreamer_model.encode(obs_t, att_t)

        action_input = old_act.unsqueeze(0)
        embed_input = enc_embed.squeeze(1)

        new_rssm_state, _ = self.rssm.observe_step(rssm_state, action_input, embed_input)

        feature = torch.cat([new_rssm_state.h, new_rssm_state.z], dim=-1)
        
        dist = self.actor(feature)

        
        action = dist.sample()

        action_numpy = action.squeeze(0).cpu().detach().numpy().astype(np.int32)
        return action_numpy, new_rssm_state
    
    def validation_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        pass
        # --
        #with torch.no_grad():
            
            #decoded_logits = self.dreamer_model.predict_obs(self.trajectory_eval)
            #video_path = Path(self.logger.log_dir).joinpath(f"train_dream_epoch_{self.current_epoch}.avi")
            #make_video(decoded_logits, video_path, fps=30)
        
    def forward(self, obs, action, user_att, mask=None, mask_actions=None, rssm_state=None):
        
        B, _, _, _, _ = obs.shape
        # -- encode attribute player with obs
        
        # -- fuse obs with user att, use simple cat fusion
        enc = self.dreamer_model.encode(obs, user_att)

        # -- store temporal informations
        if rssm_state is None:
            rssm_state = self.rssm.init_state(B, obs.device)

        
        reward        = torch.zeros(B, self.horizont, device=obs.device)
        continues     = torch.zeros(B, self.horizont, device=obs.device)
        entropys      = torch.zeros(B, self.horizont, device=obs.device)
        actions       = torch.zeros(B, self.horizont, action.shape[-1], device=obs.device)
        log_probs     = torch.zeros(B, self.horizont, device=obs.device)
        latent_states = torch.zeros(B, self.horizont, self.emb_dim, device=obs.device)
        
        # -- burn-in rssm initialization
        burn_in = self.burn_in if self.training else 0
        with torch.no_grad():
            for t in range(burn_in):
                enc_t = enc[:, t]
                action_t = action[:, t]
                m = mask[:, t]
                rssm_state, _ = self.rssm.observe_step(rssm_state, action_t, enc_t, m)
        z = rssm_state.z.detach()
        h = rssm_state.h.detach()
        z_log = rssm_state.z_logits.detach()

        rssm_state.z = z
        rssm_state.h = h 
        rssm_state.z_logits = z_log
        # -- immagination using rssm
        for t in range(self.horizont):
            embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
            embed = self.dreamer_model.feat_proj(embed)
            latent_states[:,t] =embed

            reward[:, t] = self.dreamer_model.predict_reward(embed).squeeze(-1)
            cont = (self.dreamer_model.predict_continue(embed)).squeeze(-1)
            continues[:,t] = nn.functional.sigmoid(cont)

            action, dist_a = self.actor(embed, mask_actions)

            entropys[:, t] = dist_a.entropy()
            actions[:, t] = action
            log_probs[:, t] = dist_a.log_prob(action)

            rssm_state, _ = self.rssm.imagine_step(rssm_state, action)

        
        # -- pack values
        return {
            "rewards"       : reward, 
            "continues"     : continues,
            "entropy"       : entropys, 
            "imagination"   : latent_states,
            "actions"       : actions,
            "log_probs"     : log_probs
        }

        
            


                
        
        
