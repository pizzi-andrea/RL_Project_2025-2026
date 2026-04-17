

import lightning as L
import numpy as np
import torch
import copy
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

        for param in self.target.parameters():
            param.requires_grad = False
        
        self.target_decay = 0.98
    
    def update_target(self):
       
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target.parameters()):
                target_param.data.mul_(self.target_decay).add_((1.0 - self.target_decay) * param.data)
    

   
    
   # def reparameterize_categorical(logits, tau=1.0):
   #     """Reparametrization trick for OneHot categorical sampling"""
   #     return F.gumbel_softmax(logits, tau=tau, hard=True)
    
    #def reparameterize_bernoulli(logits):
        """Reparametrization trick for Bernoullie sampling"""
    #    probs = torch.sigmoid(logits)
    #    sample = torch.bernoulli(probs)
    #    return probs + (sample - probs).detach()
    

    def configure_optimizers(self):
        # def parameters
        #   lr: 0.00003
        #   eps: 1e-20
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

        

        # Passiamo is_first al forward per il reset
        output = self.forward(obs, actions, user_att, is_first, self.burn_in,  mask, mask_a)

        # use only for target #################
        r_buffer = batch["reward"][:, self.burn_in:]
        c_buffer = batch["continue"][:, self.burn_in:]
        states_target = output["observation"]
        #######################################

       

        # -- unpack output
        states_hat = output["imagination"] # imm. step
        

        r_hat= output["rewards"]    # twohot
        c_hat = output["continues"] # linear
        #a_hat = output["actions"]
        entropy = output["entropy"]
        a_hat_log_probs = output["log_probs"] # linear
        
        

        # -- critic: estimate expected values and lambda-return
        v_hat_log_critic, v_hat_critic = self.critic(states_hat)
        v_target_log_critic, v_target = self.critic(states_target)
        

        with torch.no_grad():
            # -- target: copy of critic network use how target network to stabilize the traning
            v_hat_log_target, v_hat_target = self.target(states_hat)
            v_target_log_target, v_target_target = self.target(states_target)
           
        
            discount_c = torch.cumprod(torch.cat([torch.ones_like(c_hat[:, :1]), 0.997 * c_hat[:, :-1]], dim=1), dim=1)
            discount_t = torch.cumprod(torch.cat([torch.ones_like(c_buffer[:, :1]), 0.997 * c_buffer[:, :-1]], dim=1), dim=1)
            
            
            lambda_return = lambda_returns(
                rewards=r_hat[:, :-1],
                values=v_hat_target,
                continues=c_hat[:, :-1]
            )

            lambda_return_target = lambda_returns(
                rewards=r_buffer[:, :-1],
                values=v_target_target,
                continues=c_buffer[:, :-1]
            )

            two_hot_lambda_return_target  = twohot(lambda_return_target, self.target.buckets)
            two_hot_lambda_return= twohot(lambda_return, self.critic.buckets) 
          
            

        ###################################################################################
        
        
        # -- evaluate critic respect lambda return in twohot encoding
        critic_l = critic_loss(
            v_hat_log_critic[:, :-1],
            two_hot_lambda_return, 
            discount_c[:, :-1]
        )

        critic_l_target = critic_loss(
            v_target_log_target[:, :-1],
            two_hot_lambda_return_target, 
            discount_t[:, :-1]
        )
        

        critic_loss_v = critic_l + (0.3 * critic_l_target)
        
        # -- compute actor loss respect lambda_return and v_hat not in twohot encoding
       
    
        actor_loss_v, ema_scale, advantage, _ = actor_loss(
            lambda_return, 
            v_hat_critic[:, :-1], 
            a_hat_log_probs[:, :-1], 
            entropy[:, :-1], 
            self.ema, 
            entropy_scale=self.entropy_coef,
            continues=discount_c[:, :-1]
        )
        
        # -- ignore terminated steps

        loss = critic_loss_v + actor_loss_v

        valid_mask = discount_c[:, :-1]
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
            "critic_reply_l"  : critic_l_target.item(),
            "lambda_return" : log_lambda.mean().item(),
            "mean_reward"   : log_reward.mean().item(),
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
        self.update_target()
        obs = outputs["imagination"]
        burn = outputs["burn"]
        obs = self.dreamer_model.predict_obs(obs)
        burn = burn /255

        #obs = torch.sigmoid(obs)

        if batch_idx == 0:
            
            comparison = torch.cat([burn, obs])
            grid = torchvision.utils.make_grid(comparison, nrow=burn.shape[0])
            self.logger.experiment.add_image(
                "imagination_reconstruction", grid, self.current_epoch
            )
    
    @torch.no_grad()
    def act(self, obs, att_user, old_act, rssm_state, mask_actions=None):
        """Single step agent action"""

        embed_input = self.dreamer_model.encode(obs, att_user)
        
        new_rssm_state, _ = self.rssm.observe_step(rssm_state, old_act, embed_input)

        s = torch.cat([new_rssm_state.h, new_rssm_state.z], dim=-1)
        
        
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
        
    def forward(self, obs, action, user_att, is_first, burn_in, mask=None, mask_actions=None, rssm_state=None) :
        
        B, T, _, _, _ = obs.shape
        T = T if self.training else 0
        # -- encode attribute player with obs
        mask_first = is_first.unsqueeze(-1).float().to(self.device)
        x = self.dreamer_model.encode(obs, user_att)
        # -- fuse obs with user att, use simple cat fusion
        

        # -- store temporal informations
        if rssm_state is None:
            rssm_state = self.rssm.init_state(B, obs.device)

        
        reward        = torch.zeros(B, self.horizon, device=obs.device)
        continues     = torch.zeros(B, self.horizon, device=obs.device)
        entropys      = torch.zeros(B, self.horizon, device=obs.device)
        actions_out   = torch.zeros(B, self.horizon, action.shape[-1], device=obs.device)
        log_probs     = torch.zeros(B, self.horizon, device=obs.device)
        latent_states = torch.zeros(B, self.horizon, self.emb_dim, device=obs.device)
        
        h_rnn = torch.zeros(B, T, rssm_state.h.shape[-1], device=obs.device)
        z_rnn = torch.zeros(B, T, rssm_state.z.shape[-1], device=obs.device)
        
        last_action = torch.zeros_like(action[:, 0])
        
        
        
        
        with torch.no_grad():
            for t in range(T):
                rssm_state.h = rssm_state.h * (1.0 - mask_first)
                rssm_state.z = rssm_state.z * (1.0 - mask_first)
                
                x_t = x[:, t]
                m = mask[:, t]
                rssm_state, _ = self.rssm.observe_step(rssm_state, last_action, x_t, m)
                
                h_rnn[:, t] = rssm_state.h
                z_rnn[:, t] = rssm_state.z
                last_action = action[:, t]
        

        # coerent reset for imagination steps
            rssm_state.h = h_rnn[:, burn_in]
            rssm_state.z = z_rnn[:, burn_in]
            
            rssm_state.z = rssm_state.z * (1.0 - mask_first)
            mask_first = is_first.unsqueeze(-1).float() # Shape: [B, 1]
            
            rssm_state.h = rssm_state.h * (1.0 - mask_first)
            rssm_state.z = rssm_state.z * (1.0 - mask_first)
            
            mask_first_3d = mask_first.unsqueeze(-1)    # Shape: [B, 1, 1]
            rssm_state.z_logits = rssm_state.z_logits * (1.0 - mask_first_3d)
            
        

        # -- immagination using rssm
        s = torch.cat([rssm_state.h, rssm_state.z], dim=-1).detach()
        for t in range(self.horizon):
            
            # act in imaginated state 
            
            act_t1, log_p, entropy = self.actor(s, mask_actions)
            
            entropys[:, t] = entropy
            actions_out[:, t] = act_t1
            log_probs[:, t] = log_p
            #####################################################

            # imagine new state using only h_t + a_t
            rssm_state, _ = self.rssm.imagine_step(rssm_state, act_t1)

            s = torch.cat([rssm_state.h, rssm_state.z], dim=-1).detach()
            latent_states[:, t] = s

            

            # get reward from dreamer model
            _, r_t = self.dreamer_model.predict_reward(s)
            # get continue flag from model
            c_t = self.dreamer_model.predict_continue(s)
            
            continues[:, t] = c_t.detach()
            reward[:, t] = r_t.detach()
            #####################################################
        
        # -- pack values

        observation = torch.cat([h_rnn[:, burn_in:], z_rnn[:, burn_in:]], dim=-1)
        return {
            "rewards"       : reward, 
            "continues"     : continues,
            "entropy"       : entropys, 
            "imagination"   : latent_states,
            "actions"       : actions_out,
            "log_probs"     : log_probs,
            "observation"   : observation
            #"continue_r"    : continues_real,
        }

        
            


                
        
        
