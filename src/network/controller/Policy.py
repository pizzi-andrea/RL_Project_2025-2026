

import lightning as L
import numpy as np
import torch
import copy
from torch import nn
from timm.optim.laprop import LaProp
from torch.optim import Adam
import torchvision

from network.dreamer.rssm import RSSMState
from ..worldModel.dreamer import Dreamer
from .utils import *
from .utils import critic_loss
from .Actor import Actor
from .Critic import Critic

class DreamerAgent(L.LightningModule):
    """Agent based on Dreamer model"""
    def __init__(self, dreamer_model:Dreamer|str, 
                 buckets:int=255,
                 min_buckets:int=-20,
                 max_buckets:int=20, 
                 action_space_dim:int=16, 
                 entropy_coef:int=0.0003,
                 horizon:int=15, 
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
        self.horizon = horizon
        self.burn_in = burn_in

        # -- define EMA scaler (Mobile mean)
        self.ema = EMAScalar(ema_decay, ema_epsilon)
        
        self.emb_dim = self.dreamer_model.state_dim
        
        
        # -- Define Actor-critic Structure
        self.actor = Actor(dim_state=self.emb_dim, num_dist=action_space_dim, unimix=unimix, hidden_dim=hidden_dim)
        self.critic = Critic(dim_state=self.emb_dim, num_buckets=buckets, min_buckets=min_buckets, max_buckets=max_buckets, hidden_dim=hidden_dim)
        self.target = copy.deepcopy(self.critic)
    
    def update_target(self):
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.target.parameters()):
                p_target.data.mul_(0.98).add_(0.3* p.data)

    def configure_optimizers(self):
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        return LaProp(params, self.hparams.lr, eps=1e-8)
    
    def training_step(self, batch, batch_idx):
        """Dreamer traning procedure (burn in + dreamer)"""

        if self.preprocess_callback is not None:
            batch["obs"] = self.preprocess_callback(batch["obs"])
        
        self.ema = self.ema.train()

        # -- unpack data
        is_first = batch["is_first"] # [B]
        
        obs = batch["obs"]
        actions = batch["action"]
        user_att = batch["user_att"]
        
        mask = batch["mask"]
        mask_a = batch["mask_action"]

        burning = min(self.burn_in, obs.shape[1] // 2)

        output = self.forward(obs, actions, user_att, is_first, burning,  mask, mask_a)

        # use only for target #################
        r_real = batch["reward"]
        continue_real = batch["continue"]
        states_real = output["real_states"]

        #######################################

       

        # -- unpack output
        states_hat = output["imagination"] # imm. step
        
        r_hat= output["rewards"]    # linear space
        c_hat = output["continues"] # linear
        entropy = output["entropy"]
        a_hat_log_probs = output["log_probs"] # linear
        
        

        # -- critic: estimate expected values and lambda-return
        v_hat_log_critic, v_hat_critic = self.critic(states_hat)
        v_hat_log_critic_real, v_real = self.critic(states_real)
        
        with torch.no_grad():
            # -- target: copy of critic network use how target network to stabilize the traning
            v_hat_log_target, v_hat_target = self.target(states_hat)
            v_hat_log_target_real, v_hat_target_real = self.target(states_real)
            
            
           
            discount_c = torch.cumprod(
                torch.cat([
                    torch.ones_like(c_hat[:, :1]),
                    0.997 * c_hat[:, :-1]          # ← :-1 invece del tensore intero
                ], dim=1),
                dim=1
            )  # [B, H], niente slice finale necessari

            discount_r = torch.cumprod(
                torch.cat([
                    torch.ones_like(continue_real[:, :1]),
                    0.997 * continue_real[:, :-1]          # ← :-1 invece del tensore intero
                ], dim=1),
                dim=1
            ) # [B, H], niente slice finale necessari
            
            lambda_return = lambda_returns(
                rewards=r_hat,
                expected_values=v_hat_target,
                continues=c_hat, 
            )

            lambda_return_real = lambda_returns(
                rewards=r_real,
                expected_values=v_hat_target_real,
                continues=continue_real
            )

         
            two_hot_lambda_return= twohot(symlog(lambda_return), self.critic.buckets)
            two_hot_lambda_return_real =  twohot(symlog(lambda_return_real), self.critic.buckets)

          
        ###################################################################################
        
        
        # -- evaluate critic respect lambda return in twohot encoding
        # actor loss: advantage = lambda_return - V(s_t), entrambi [B, H]
        
        actor_loss_v, ema_scale, advantage, _ = actor_loss(
            lambda_return,
            v_hat_critic[:, :-1].detach(),
            a_hat_log_probs,
            entropy,
            self.ema,
            entropy_scale=self.entropy_coef,
            c_soft=discount_c,
            c_hard=c_hat
        )

        
        critic_l = critic_loss(
            v_hat_log_critic[:, :-1],     # [B, H, buckets] ✓
            two_hot_lambda_return,        # [B, H, buckets] ✓
            discount_c
        )

        critic_r = critic_loss(
            v_hat_log_critic_real[:, :-1],
            two_hot_lambda_return_real,
            discount_r

        )

       
        
        
        critic_loss_v = critic_l + 0.3 * critic_r

        loss = critic_loss_v + actor_loss_v

        valid_mask = discount_c
        valid_mask_r = discount_r
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        lengths_r = valid_mask_r.sum(dim=1).clamp(min=1)

        with torch.no_grad():
            # Lambda return medio per traiettoria, poi media del batch
            log_lambda = lambda_return.sum(dim=1) / lengths
            
            # Reward media (attenzione: r_hat qui deve essere il valore scalare, non two-hot)
            # Se r_hat è scalare (es. dopo symexp o decoding):
            log_reward = (r_hat * valid_mask).sum(dim=1) / lengths
            real_reward = (r_real * discount_r).sum(dim=1) / lengths_r
            real_lambda = (lambda_return_real).sum(dim=1) / lengths_r
            
            # Entropia media
            log_entropy = entropy

   
        self.log_dict({
            "train_loss"    : loss.item(),
            "actor_loss"    : actor_loss_v.item(),
            "critic_loss"   : critic_loss_v.item(),
            "lambda_return" : log_lambda.mean().item(),
            "mean_reward"   : log_reward.mean().item(),
            "real_reward"   : real_reward.mean().item(),
            "real_lambda"   : real_lambda.mean().item(),
            "entropy"       :log_entropy.mean().item(),
            "ema_scale"     : ema_scale,
            "advantage"     : advantage
        }, prog_bar=True, on_step=True, on_epoch=False)

        return {
            
            "loss": loss,
            "imagination": output["imagination"][0],
            "burn": obs[0]
        }
    
    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.eval()
        obs = outputs["imagination"]
        real = outputs["burn"]
        self.update_target()
    
        obs = self.dreamer_model.predict_obs(obs)
        real = (real / 255)

        burning = real[:self.burn_in]
        end_frame = torch.zeros_like(burning[0]).unsqueeze(0)
        if batch_idx == 0:
            
            comparison = torch.cat([burning, end_frame, obs], dim=0)
            
            grid = torchvision.utils.make_grid(
                comparison, 
                nrow=comparison.shape[0],
                )
            self.logger.experiment.add_image(
                "imagination", grid, self.global_step
            )
    def configure_gradient_clipping(self, optimizer, gradient_clip_val = None, gradient_clip_algorithm = None):
        AGC(self.parameters(), clip_factor=0.3)
    @torch.no_grad()
    def act(self, obs, att_user, old_act, rssm_state, mask_actions=None):
        """Single step agent action"""

        embed_input, user_att = self.dreamer_model.encode(obs, att_user)
        a_t = self.dreamer_model.encode_action(old_act)
        new_rssm_state, _, _ = self.rssm.observe_step(rssm_state, a_t, embed_input, user_att)

        z_flat = new_rssm_state.z.flatten(1)
        s = self.dreamer_model.encode_state(new_rssm_state.h, z_flat)
        
        
        action, _, _ = self.actor(s, mask_actions)


        action_numpy = action.squeeze(0).cpu().detach().numpy().astype(np.uint8)
        return action_numpy, new_rssm_state
    
    def validation_step(self, batch, batch_idx):
        pass

        
        # --
        #with torch.no_grad():
            
            #decoded_logits = self.dreamer_model.predict_obs(self.trajectory_eval)
            #video_path = Path(self.logger.log_dir).joinpath(f"train_dream_epoch_{self.current_epoch}.avi")
            #make_video(decoded_logits, video_path, fps=30)
        
    def forward(self, obs, action, user_att, is_first, burn_in:int, mask=None, mask_actions=None, rssm_state=None):

        B, T, _, _, _ = obs.shape

        x, user_att = self.dreamer_model.encode(obs, user_att)

        x = x.detach()
        user_att = user_att.detach()

        if rssm_state is None:
            rssm_state = self.rssm.init_state(B, obs.device)

        reward        = torch.empty(B, self.horizon, device=obs.device)
        continues     = torch.empty(B, self.horizon, device=obs.device)
        entropys      = torch.empty(B, self.horizon, device=obs.device)
        log_probs     = torch.empty(B, self.horizon, device=obs.device)
        latent_states = torch.empty(B, self.horizon, self.emb_dim, device=obs.device)

        real_states        = torch.empty(B, T, self.emb_dim, device=obs.device)
        last_action = torch.zeros_like(action[:, 0])

        burning_state = torch.empty(B, self.emb_dim, device=obs.device)
        burning_last_action = torch.zeros_like(action[:, 0])

        with torch.no_grad():
            for t in range(T):

                a_t = self.dreamer_model.encode_action(last_action)
                
                rssm_state, _, _ = self.rssm.observe_step(
                    rssm_state,
                    a_t,
                    x[:, t],
                    user_att[:, t],
                    mask[:, t]
                )
                z_flat = rssm_state.z.flatten(1)
                h      = rssm_state.h
                s      = self.dreamer_model.encode_state(h, z_flat)
               
                last_action = action[:, t]

                real_states[:, t] = s 

                if t + 1 == burn_in:
                    burning_rssm_state, last_action_ = self._apply_episode_reset(rssm_state, last_action, is_first)
                    
                    z_flat = burning_rssm_state.z.flatten(1)
                    
                    h      = burning_rssm_state.h
                    s      = self.dreamer_model.encode_state(h, z_flat)
                    burning_state = s
                    burning_last_action = last_action_
        

                if burn_in == 0:
                    rssm_state_, last_action_ = self._apply_episode_reset(
                        rssm_state, last_action, is_first
                    )
                    burning_rssm_state  = rssm_state_          # ✓
                    burning_last_action = last_action_
                    burning_state = self.dreamer_model.encode_state(
                        rssm_state_.h, rssm_state_.z.flatten(1)
                    )

        real_states = torch.cat([real_states, real_states[:, -1:]], dim=1)  # ✓ [B, T+1]

        s           = burning_state
        last_action = burning_last_action
        rssm_state  = burning_rssm_state   

        

        for t in range(self.horizon):
            act_t1, log_p, entropy = self.actor(s, mask_actions)
            latent_states[:, t] = s
            
            _, r_t = self.dreamer_model.predict_reward(s)
            c_t    = self.dreamer_model.predict_continue(s)
            user_att_t = self.dreamer_model.predict_user_att(s).detach()


            reward[:, t]    = r_t.detach()
            continues[:, t] = c_t.detach()
            #user_att[:, t]  = user_att

       
            log_probs[:, t]   = log_p
            entropys[:, t]    = entropy
                        
            act_t1, user_att_t = self.dreamer_model.encode_action(act_t1), self.dreamer_model.encode_att(user_att_t)
            rssm_state, _ = self.rssm.imagine_step(rssm_state, act_t1, user_att_t)
            
            z_flat = rssm_state.z.flatten(1).detach()
            
            h      = rssm_state.h.detach()
            s      = self.dreamer_model.encode_state(h, z_flat).detach()  # s = z_{t+1}
        
        latent_states = torch.cat([latent_states, s.unsqueeze(1)], dim=1)  # [B, H+1, emb]

        return {
            "rewards": reward,
            "continues": continues,
            "entropy": entropys,
            "imagination": latent_states,
            "log_probs": log_probs,
            "real_states"    : real_states
           
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

        state.h        = torch.where(first.bool(),  fresh.h, state.h)
        state.z        = torch.where(first3.bool(), fresh.z, state.z)

        last_action    = torch.where(first.bool(), torch.zeros_like(last_action), last_action)

        return state, last_action            

