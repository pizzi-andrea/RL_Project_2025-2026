from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor, nn
from timm.optim.laprop import LaProp

from ..world.Dreamer import Dreamer
from .utils import *
from .Actor import Actor
from .Critic import Critic

class DreamerAgent(L.LightningModule):
    def __init__(self, dreamer_model:Dreamer|str, bukets=255, action_space_dim:int=16, entropy_coef:int=0.0003,horizont:int=15, burn_in:int=3, lr=0.001, one_hot:bool=True):
        super().__init__()
        self.save_hyperparameters(ignore=["dreamer_model"])
        # import dreamer model or weigths
        if isinstance(dreamer_model, str):
            self.dreamer_model = Dreamer.load_from_checkpoint(dreamer_model)
        else:
            self.dreamer_model = dreamer_model
        
        

        # -- obtain RSSM model
        self.rssm = self.dreamer_model.rssm
        self.entropy_coef = entropy_coef
        self.ema = EMAScalar()
        self.emb_dim = self.dreamer_model.internal_hidden_dim
        self.action_space_dim = action_space_dim
        self.horizont = horizont
        self.burn_in = burn_in
        
        
        self.actor = Actor(dim_state=self.emb_dim, num_dist=action_space_dim)
        self.critic = Critic(dim_state=self.emb_dim, num_buckets=bukets)

    def setup(self, stage):
        
        self.dreamer_model.freeze_model()
    
    def reparameterize_categorical(logits, tau=1.0):
        
        return F.gumbel_softmax(logits, tau=tau, hard=True)
    
    def reparameterize_bernoulli(logits):
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs)
        return probs + (sample - probs).detach()
    

    def configure_optimizers(self):
        return LaProp(self.parameters(), lr=self.hparams.lr, eps=1e-6)

    def training_step(self, batch, batch_idx):
        # -- 
        obs     = batch["obs"][:, 0:self.burn_in]
        actions = batch["action"][:, 0:self.burn_in]
        user_att = batch["user_att"][:, 0:self.burn_in]
        mask     = batch["mask"][:, 0:self.burn_in]
        mask_a   = batch["mask_action"]

        
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0).detach()

        output = self.forward(obs, actions, user_att, mask, mask_a)

        # -- 
        states_hat = output["imagination"]
        r_hat = output["rewards"]
        c_hat = output["continues"]
        a_hat = output["actions"]
        entropy = output["entropy"]
        log_probs = output["log_probs"]
        

        #if self.trainer.num_training_batches == batch_idx + 1:
        #    self.trajectory_eval = states_hat[0, :].cpu().detach()
        critic_log = self.critic(states_hat)

        with torch.no_grad():
            critic_prob = torch.softmax(critic_log, dim=-1)
            value_symlog = (critic_prob * self.critic.buckets).sum(-1)
            value_symexp = torch.sign(value_symlog) * (torch.exp(torch.abs(value_symlog)) - 1)
        
        td_error = lambda_returns(
            rewards=r_hat,
            values=value_symexp,
            continues=c_hat
        )
        actions = torch.round(a_hat).detach()
        
        critic_l = critic_loss(critic_log[:, :-1], td_error, buckets=self.critic.num_buckets)
        #print(critic_log.shape)
        #print(td_error.shape)
        #print(value_symexp[:, :-1].shape)
        #exit()
        actor_l = actor_loss(td_error, value_symexp[:, :-1], log_probs[:, :-1], entropy[:, :-1], self.ema, entropy_scale=self.entropy_coef)
        loss = critic_l + actor_l
        
        self.log_dict({
            "train_loss"    : loss.detach().item(),
            "actor_loss"    : actor_l.detach().item(),
            "critic_loss"   : critic_l.detach().item(),
            "lambda_return" : td_error.mean().item(),
            "mean_reward"   : r_hat.detach().mean().item(),
            "entropy"       : entropy.detach().mean().item()
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
        with torch.no_grad():
            for t in range(self.burn_in):
                enc_t = enc[:, t]
                action_t = action[:, t]
                m = mask[:, t]
                rssm_state, _ = self.rssm.observe_step(rssm_state, action_t, enc_t, m)
            
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

        
            


                
        
        
