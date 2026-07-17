

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
                 lambd:float=0.95, 
                 gamma:float=0.997,
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
        self.lambd = lambd
        self.gamma = gamma
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
                p_target.data.mul_(0.98).add_(0.02 * p.data)

    def configure_optimizers(self):
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        return LaProp(params, self.hparams.lr, eps=1e-20)
    
    def on_train_start(self):
        self.dreamer_model.freeze()
        self.rssm.freeze()

    def training_step(self, batch, batch_idx):
        """Dreamer training procedure (burn in + dreamer)"""

        if self.preprocess_callback is not None:
            batch["obs"] = self.preprocess_callback(batch["obs"])
        
        self.ema = self.ema.train()

        # -- unpack data
        is_first = batch["is_first"] 
        obs = batch["obs"]
        actions = batch["action"]
        user_att = batch["user_att"]
        mask = batch["mask"]

        # Burn-in dinamico: garantiamo che non superi metà sequenza
        burning = min(self.burn_in, obs.shape[1] // 2)

        # Il forward ora gestisce internamente il Masked Burn-in e il Reset
        output = self.forward(obs, actions, user_att, is_first, burning, mask)

        # Dati reali per target e critic loss
        r_real = batch["reward"]
        continue_real = batch["continue"]
        states_real = output["real_states"] # [B, T+1, dim]

        # -- unpack output immaginazione
        states_hat = output["imagination"] 
        r_hat = output["rewards"]    
        c_hat = output["continues"] 
        entropy = output["entropy"]
        a_hat_log_probs = output["log_probs"] 

        # -- critic: stima valori
        v_hat_log_critic, v_hat_critic = self.critic(states_hat)
        
        # Per i real states, states_real è T+1, quindi v_hat_log_critic_real sarà [B, T+1, buckets]
        v_hat_log_critic_real, v_hat_critic_real = self.critic(states_real)
        
        with torch.no_grad():
            v_hat_log_target, v_hat_target = self.target(states_hat)
            v_hat_log_target_real, v_hat_target_real = self.target(states_real)
            
            # --- DEFINIZIONE MASCHERE ---
            # 1. Maschere per dati REALI (Hanno padding derivato dal batch)
            valid_mask_real = mask[:, :r_real.shape[1]] 
            
            exec_mask_real = continue_real * valid_mask_real
            exec_mask_real[:, -1] = 0
            
            # 2. Maschere per dati IMMAGINATI (Non hanno padding, orizzonte fisso)
            valid_mask_imag = torch.ones_like(c_hat)
            exec_mask_imag = c_hat * valid_mask_imag # Coincide con c_hat

            # --- SCONTI TEMPORALI (Opzionali ma utili in Dreamer) ---
            # Combinano lo sconto temporale con la validità del frame
            discount_c = torch.cumprod(
                torch.cat([torch.ones_like(c_hat[:, :1]), self.gamma * c_hat[:, :-1]], dim=1), dim=1
            ) * valid_mask_imag

            discount_r = torch.cumprod(
                torch.cat([torch.ones_like(continue_real[:, :1]), self.gamma * continue_real[:, :-1]], dim=1), dim=1
            ) * valid_mask_real 

            # --- LAMBDA RETURNS (Uso di execution_mask) ---
            lambda_return = lambda_returns(
                rewards=r_hat,
                expected_values=v_hat_target,
                execution_mask=exec_mask_imag,
                lam=self.lambd,
                gamma=self.gamma
            )

            lambda_return_real = lambda_returns(
                rewards=r_real,
                expected_values=v_hat_target_real,
                execution_mask=exec_mask_real,
                lam=self.lambd,
                gamma=self.gamma
            )

            two_hot_lambda_return = twohot(symlog(lambda_return), self.critic.buckets)
            two_hot_lambda_return_real = twohot(symlog(lambda_return_real), self.critic.buckets)

        # -- Actor Loss (Uso di valid_mask / discount)
        actor_loss_v, ema_scale, advantage, _ = actor_loss(
            td_return=lambda_return,
            critic_values=v_hat_critic[:, :-1].detach(),
            action_log_prob=a_hat_log_probs,
            entropy=entropy,
            ema_scaler=self.ema,
            entropy_scale=self.entropy_coef,
            c_soft=discount_c, # Maschera di validità con decadimento
            mask=valid_mask_imag
        )

        # -- Critic Loss (Uso di valid_mask / discount)
        critic_l = critic_loss(
            v_hat_log_critic[:, :-1],     
            two_hot_lambda_return,        
            discount_c 
        )

        critic_r = critic_loss(
            v_hat_log_critic_real[:, :-1],
            two_hot_lambda_return_real,
            discount_r 
        )
        
        critic_loss_v = critic_l + 0.3 * critic_r
        loss = critic_loss_v + actor_loss_v

        # -- Metrics
        lengths = discount_c.sum(dim=1).clamp(min=1)
        lengths_r = discount_r.sum(dim=1).clamp(min=1)

        with torch.no_grad():
            log_lambda = (lambda_return * discount_c).sum(dim=1) / lengths
            log_reward = (r_hat * discount_c).sum(dim=1) / lengths
            real_reward = (r_real * discount_r).sum(dim=1) / lengths_r
            real_lambda = (lambda_return_real * discount_r).sum(dim=1) / lengths_r
            entropy_m = ((entropy * discount_c).sum(dim=1) / lengths).mean()
            
     

        self.log_dict({
            "entropy"       : entropy_m.item(),
            "lambda_real"   : real_lambda.mean().item(),
            "lambda_return" : log_lambda.mean().item(),
            "train_loss"    : loss.item(),
            "actor_loss"    : actor_loss_v.item(),
            "critic_loss"   : critic_loss_v.item(),
            "mean_reward"   : log_reward.mean().item(),
            "real_reward"   : real_reward.mean().item(),
            "ema_scale" : ema_scale,
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
    
        imagined_frames = torch.clamp(self.dreamer_model.predict_obs(obs), 0.0, 1.0)

        if  self.global_step % 100 == 0:
            log_dir = self.logger.log_dir if (self.logger and hasattr(self.logger, 'log_dir')) else "."
            make_video(imagined_frames, f"{log_dir}/imagined_{self.global_step}.avi", fps=10)
            print(f"new video save in {log_dir}")

        if batch_idx == 0 and self.logger is not None:
            burning = (real[:self.burn_in] / 255)
            comparison = torch.cat([burning, torch.zeros_like(burning[0]).unsqueeze(0), imagined_frames], dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=comparison.shape[0])
            self.logger.experiment.add_image("imagination_static_grid", grid, self.global_step)
   
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
        
    def forward(self, obs, action, user_att, is_first, burn_in:int, mask=None, rssm_state=None):

        B, T, _, _, _ = obs.shape
        device = obs.device

        # 1. Encoding (World Model frozen)
        x, user_att_enc = self.dreamer_model.encode(obs, user_att)
        x = x.detach()
        user_att_enc = user_att_enc.detach()

        if rssm_state is None:
            rssm_state = self.rssm.init_state(B, device)

        # Buffer per Critic (Real) e Actor (Imagined)
        real_states = torch.empty(B, T, self.emb_dim, device=device)
        last_action = torch.zeros_like(action[:, 0])

        # --- FASE 1: OBSERVATION (Burn-in con Maschera) ---
        # Questa fase serve a portare l'RSSM allo stato attuale "reale" 
        # rispettando il padding variabile.
        with torch.no_grad():
            for t in range(T):
                # Maschera al tempo t: 1 = info, 0 = padding
                m_t = mask[:, t].unsqueeze(-1)
                
                a_t = self.dreamer_model.encode_action(last_action)
                
                # Calcolo step successivo
                next_rssm_state, _, _ = self.rssm.observe_step(
                    rssm_state, a_t, x[:, t], user_att_enc[:, t], mask[:, t]
                )

                # 🔒 MASKED UPDATE: Se m_t è 0, congeliamo lo stato precedente
                rssm_state.h = torch.where(m_t.bool(), next_rssm_state.h, rssm_state.h)
                
                m_t_z = m_t.unsqueeze(-1) # [B, 1, 1]
                rssm_state.z = torch.where(m_t_z.bool(), next_rssm_state.z, rssm_state.z)
                
                # Aggiorniamo l'azione per lo step successivo solo se valida
                last_action = torch.where(m_t.bool(), action[:, t], last_action)

                # Salviamo lo stato reale (per il Critic Loss r)
                z_flat = rssm_state.z.flatten(1)
                real_states[:, t] = self.dreamer_model.encode_state(rssm_state.h, z_flat)

                # 🎯 CATTURA START POINT PER IMMAGINAZIONE
                # Salviamo lo stato esattamente alla fine della finestra di burn_in
                if t + 1 == burn_in:
                    # Applichiamo il reset: se is_first è True, lo stato accumulato
                    # finora viene azzerato (inizio episodio pulito).
                    burning_rssm_state, last_action_imag = self._apply_episode_reset(
                        rssm_state, last_action, is_first
                    )

                    current_imag_rssm_state = RSSMState(
                        h=burning_rssm_state.h.clone(),
                        z=burning_rssm_state.z.clone()
                    )
                    
                    
                    
                    z_flat_imag = current_imag_rssm_state.z.flatten(1)
                    current_imag_s = self.dreamer_model.encode_state(current_imag_rssm_state.h, z_flat_imag)
                    last_action_imag = last_action_imag.clone()
                    

            # Caso limite: burn_in a 0
            if burn_in == 0:
                rssm_state_reset, action_reset = self._apply_episode_reset(
                    self.rssm.init_state(B, device), last_action, is_first
                )
                current_imag_rssm_state = rssm_state_reset
                current_imag_s = self.dreamer_model.encode_state(
                    rssm_state_reset.h, rssm_state_reset.z.flatten(1)
                )
                last_action_imag = action_reset
        # Estendiamo i real_states per allineamento critic
        real_states = torch.cat([real_states, real_states[:, -1:]], dim=1)

        # --- FASE 2: IMAGINATION (Policy Training) ---
        reward        = torch.empty(B, self.horizon, device=device)
        continues     = torch.empty(B, self.horizon, device=device)
        entropys      = torch.empty(B, self.horizon, device=device)
        log_probs     = torch.empty(B, self.horizon, device=device)
        latent_states = torch.empty(B, self.horizon, self.emb_dim, device=device)

        s           = current_imag_s
        rssm_state  = current_imag_rssm_state
        last_action = last_action_imag
        # Nota: in immaginazione non usiamo l'azione reale, ma quella dell'Actor

        for t in range(self.horizon):
            # L'Actor sceglie l'azione basandosi sullo stato s
            act_t1, log_p, entropy = self.actor(s)

            latent_states[:, t] = s

            # Predizioni del World Model (Frozen)
            _, r_t = self.dreamer_model.predict_reward(s)
            c_t    = self.dreamer_model.predict_continue(s)
            user_att_t_hat = self.dreamer_model.predict_user_att(s)

            reward[:, t]    = r_t
            continues[:, t] = c_t
            log_probs[:, t] = log_p
            entropys[:, t]  = entropy

            # Transizione immaginaria (Senza osservazione reale)
            act_t1_enc = self.dreamer_model.encode_action(act_t1)
            att_t1_enc = self.dreamer_model.encode_att(user_att_t_hat)

            rssm_state, _ = self.rssm.imagine_step(rssm_state, act_t1_enc, att_t1_enc)
            
            z_flat = rssm_state.z.flatten(1)
            s = self.dreamer_model.encode_state(rssm_state.h, z_flat)

        latent_states = torch.cat([latent_states, s.unsqueeze(1)], dim=1)

        return {
            "rewards": reward,
            "continues": continues,
            "entropy": entropys,
            "imagination": latent_states,
            "log_probs": log_probs,
            "real_states": real_states
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
        if is_first is None:
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

