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
from .losses import *
from .Actor import Actor
from .Critic import Critic

class DreamerAgent(L.LightningModule):
    """
    Actor-Critic agent based on REINFORCE algorithm with robust critic policy estimator
    Args:
        
    """
    def __init__(self, dreamer_model: Dreamer | str, 
                 buckets: int = 255,
                 min_buckets: int = -20,
                 max_buckets: int = 20, 
                 action_space_dim: int = 16, 
                 entropy_coef: float = 0.0003,
                 horizon: int = 15, 
                 burn_in: int = 3, 
                 unimix: float = 0.01, 
                 critic_ema_decay: float = 0.98,   
                 retnorm_decay: float = 0.99,      
                 retnorm_limit: float = 1.0,       
                 lambd: float = 0.95, 
                 gamma: float = 0.997,
                 lr_actor: float = 4e-5,    
                 lr_critic: float = 4e-5,
                 hidden_dim: int = 1024,
                 preprocess_callback=None):
        super().__init__()
        self.automatic_optimization = False 
        self.save_hyperparameters(ignore=['dreamer_model', 'preprocess_callback'])
        
        if isinstance(dreamer_model, str):
            self.dreamer_model = Dreamer.load_from_checkpoint(dreamer_model)
        else:
            self.dreamer_model = dreamer_model
        
        self.preprocess_callback = preprocess_callback

        self.rssm = self.dreamer_model.rssm
        self.entropy_coef = entropy_coef
        self.action_space_dim = action_space_dim
        self.horizon = horizon
        self.burn_in = burn_in
        self.lambd = lambd
        self.gamma = gamma
        
        # prepare EMA scaler
        self.ema = EMAScalar(decay=retnorm_decay, eps=retnorm_limit)
        
        self.emb_dim = self.dreamer_model.state_dim
        
        self.actor = Actor(dim_state=self.emb_dim, num_dist=action_space_dim, unimix=unimix, hidden_dim=hidden_dim)
        self.critic = Critic(dim_state=self.emb_dim, num_buckets=buckets, min_buckets=min_buckets, max_buckets=max_buckets, hidden_dim=hidden_dim)
        self.target = copy.deepcopy(self.critic)
    
    def update_target(self):
        with torch.no_grad():
            decay = self.hparams.critic_ema_decay
            for p, p_target in zip(self.critic.parameters(), self.target.parameters()):
                p_target.data.mul_(decay).add_((1.0 - decay) * p.data)

    def configure_optimizers(self):
        opt_actor = LaProp(self.actor.parameters(), lr=self.hparams.lr_actor, eps=1e-8)
        opt_critic = LaProp(self.critic.parameters(), lr=self.hparams.lr_critic, eps=1e-8)
        return [opt_actor, opt_critic]
    
    def on_train_start(self):
        self.dreamer_model.freeze()
        self.rssm.freeze()

    def training_step(self, batch, batch_idx):
        if self.preprocess_callback is not None:
            batch["obs"] = self.preprocess_callback(batch["obs"])
        
        self.ema = self.ema.train()

        is_first = batch["is_first"] 
        obs = batch["obs"]
        actions = batch["action"]
        user_att = batch["user_att"]
        mask = batch["mask"]

        burning = min(self.burn_in, obs.shape[1] // 2)
        valid_actions_mask = batch.get("valid_actions_mask", None)
        output = self.forward(obs, actions, user_att, is_first, burning, mask, valid_actions_mask)

        r_real = batch["reward"]
        continue_real = batch["continue"]
        states_real = output["real_states"] 

        states_hat = output["imagination"] 
        r_hat = output["rewards"]    
        c_hat = output["continues"] 
        entropy = output["entropy"]
        a_hat_log_probs = output["log_probs"] 

    
        with torch.no_grad():
            # obtain value in linear space for lambda return 
            v_hat_target_linear = self.critic.predict_value(states_hat)
            v_hat_target_real_linear = self.critic.predict_value(states_real)

            lambda_return = lambda_returns(
                rewards=r_hat,
                expected_values=v_hat_target_linear, 
                execution_mask=c_hat,
                lam=self.lambd,
                gamma=self.gamma
            )

            lambda_return_real = lambda_returns(
                rewards=r_real,
                expected_values=v_hat_target_real_linear,
                execution_mask=continue_real,
                lam=self.lambd,
                gamma=self.gamma
            )

            
            # transform values in symlog space for loss 
            two_hot_lambda_return = twohot(symlog(lambda_return), self.critic.buckets)
            two_hot_lambda_return_real = twohot(symlog(lambda_return_real), self.critic.buckets)

        # === 2. Actor Loss (Baseline nello Spazio Lineare) ===
        

        v_hat_critic_linear = self.critic.predict_value(states_hat[:, :-1]).detach()
        
        actor_loss_v, ema_scale, advantage, _ = actor_loss(
            td_return=lambda_return, 
            critic_values=v_hat_critic_linear, 
            action_log_prob=a_hat_log_probs,
            entropy=entropy,
            ema_scaler=self.ema,
            entropy_scale=self.entropy_coef,
            c_soft=c_hat, 
            mask=c_hat    
        )

        v_hat_log_critic = self.critic.decode_value(states_hat[:, :-1])
        v_hat_log_critic_real = self.critic.decode_value(states_real[:, :-1])

      
        critic_l = critic_loss(v_hat_log_critic, two_hot_lambda_return, c_hat)
        critic_r = critic_loss(v_hat_log_critic_real, two_hot_lambda_return_real, continue_real)
        
        # complex regulation need to stabilize mobile target learning 
        weight_reg = 0.0
        for p, p_target in zip(self.critic.parameters(), self.target.parameters()):
            weight_reg += torch.sum((p - p_target.detach()) ** 2)
        
        critic_loss_v = critic_l + 0.3 * critic_r + 0.01 * weight_reg
        loss = critic_loss_v + actor_loss_v

        # == OPTIMIZATION ============================================================
        opt_actor, opt_critic = self.optimizers()
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        
        self.manual_backward(loss)

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        
        opt_actor.step()
        opt_critic.step()
        # ===============================================================================
        
        # == Metrics ====================================================================
        lengths = c_hat.sum(dim=1).clamp(min=1)
        lengths_r = continue_real.sum(dim=1).clamp(min=1)

        with torch.no_grad():
            symlog_lambda_real = (symlog(lambda_return_real) * continue_real).sum(dim=1) / lengths_r
            log_lambda = (lambda_return * c_hat).sum(dim=1) / lengths
            log_reward = (r_hat * c_hat).sum(dim=1) / lengths
            real_reward = (r_real * continue_real).sum(dim=1)
            real_lambda = (lambda_return_real * continue_real).sum(dim=1) / lengths_r
            entropy_m = ((entropy * c_hat).sum(dim=1) / lengths).mean()

        self.log_dict({
            "entropy"       : entropy_m.item(),
            "lambda_real_symlog": symlog_lambda_real.mean().item(),
            "lambda_return" : log_lambda.mean().item(),
            "train_loss"    : loss.item(),
            "actor_loss"    : actor_loss_v.item(),
            "critic_loss"   : critic_loss_v.item(),
            "mean_reward"   : log_reward.mean().item(),
            "real_reward"   : real_reward.mean().item(),
            "ema_scale"     : ema_scale,
            "advantage"     : advantage
        }, prog_bar=True, on_step=True, on_epoch=False)

        return {
            "loss": loss,
            "imagination": output["imagination"].detach(), 
            "burn": obs.detach(),
            "len_r": lengths_r,
            "len_hat": lengths
        }
    
    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.eval()
        self.update_target()
        
        
        len_hat = int(outputs["len_hat"].item()) if torch.is_tensor(outputs["len_hat"]) else int(outputs["len_hat"])
        len_r = int(outputs["len_r"].item()) if torch.is_tensor(outputs["len_r"]) else int(outputs["len_r"])
        
        
        latent_space_sample = outputs["imagination"][0].unsqueeze(0) 
        imagined_frames = self.dreamer_model.predict_obs(latent_space_sample[:, :len_hat - 1, :, :])
        imagined_frames = imagined_frames.squeeze(0)  # Output atteso: [Time, Canali, Altezza, Larghezza]

        
        real = outputs["burn"][0] 
        real = real[:len_r - 1, :, :, :] # (4D: T, C, H, W)

       
        if self.global_step % 100 == 0:
            log_dir = self.logger.log_dir if (self.logger and hasattr(self.logger, 'log_dir')) else "."
            make_video(imagined_frames, f"{log_dir}/imagined_{self.global_step}.avi", fps=10)
            print(f"new video save in {log_dir}")


        if batch_idx == 0 and self.logger is not None:
            burning_steps = min(self.burn_in, real.shape[0])
            burning = real[:burning_steps]
            
       
            if burning.dtype == torch.uint8:
                burning = burning.float() / 255.0
            if imagined_frames.dtype == torch.uint8:
                imagined_frames = imagined_frames.float() / 255.0
                
          
            black_separator = torch.zeros_like(burning[0]).unsqueeze(0)
          
            comparison = torch.cat([burning, black_separator, imagined_frames], dim=0)
            
            grid = torchvision.utils.make_grid(comparison, nrow=comparison.shape[0])
            self.logger.experiment.add_image("imagination_static_grid", grid, self.global_step)
   
    @torch.no_grad()
    def act(self, obs, att_user, old_act, rssm_state, mask_actions=None):
        embed_input = self.dreamer_model.encode_obs(obs)
        user_att = self.dreamer_model.encode_att(att_user)
        a_t = self.dreamer_model.encode_action(old_act)
        new_rssm_state, _, _ = self.rssm.observe_step(rssm_state, a_t, embed_input, user_att)

        z_flat = new_rssm_state.z.flatten(1)
        s = self.dreamer_model.encode_state(new_rssm_state.h, z_flat)
        
        action, _, _ = self.actor(s, mask_actions)
        action_numpy = action.squeeze(0).cpu().detach().numpy().astype(np.uint8)
        return action_numpy, new_rssm_state
    
    def validation_step(self, batch, batch_idx):
        pass
        
    def forward(self, obs, action, user_att, is_first, burn_in:int, mask=None, valid_actions_mask=None, rssm_state=None):
            B, T, _, _, _ = obs.shape
            device = obs.device

            x = self.dreamer_model.encode_obs(obs)
            user_att_enc = self.dreamer_model.encode_att(user_att)
            x = x.detach()
            user_att_enc = user_att_enc.detach()

            # Inizializzazione pulita
            if rssm_state is None:
                rssm_state = self.rssm.init_state(B, device)

            last_action = action[:, 0].clone() 

            # === 1. RESET INIZIALE CORRETTO ===
            # Resettiamo la memoria SOLO all'istante t=0 per i nuovi episodi. 
            # In questo modo il burn-in costruirà un ricordo solido e non verrà cancellato.
            rssm_state, last_action = self._apply_episode_reset(
                rssm_state, last_action, action[:, 0], is_first
            )

            real_states = torch.empty(B, T, self.emb_dim, device=device)
            
            # Buffer per salvare l'intero storico degli stati interni dell'RSSM
            h_history = torch.empty(B, T, self.rssm.h_dim, device=device)
            z_history = torch.empty(B, T, self.rssm.L, self.rssm.K, device=device)

            # === 2. OSSERVAZIONE DELLA REALTÀ (TUTTA LA SEQUENZA) ===
            with torch.no_grad():
                for t in range(T):
                    m_t = mask[:, t].unsqueeze(-1)
                    a_t = self.dreamer_model.encode_action(last_action)
                    
                    next_rssm_state, _, _ = self.rssm.observe_step(
                        rssm_state, a_t, x[:, t], user_att_enc[:, t], mask[:, t]
                    )

                    rssm_state.h = torch.where(m_t.bool(), next_rssm_state.h, rssm_state.h)
                    
                    m_t_z = m_t.unsqueeze(-1) 
                    rssm_state.z = torch.where(m_t_z.bool(), next_rssm_state.z, rssm_state.z)
                    
                    last_action = torch.where(m_t.bool(), action[:, t], last_action)

                    z_flat = rssm_state.z.flatten(1)
                    real_states[:, t] = self.dreamer_model.encode_state(rssm_state.h, z_flat)
                    
                    # Salviamo lo stato esatto a questo istante temporale
                    h_history[:, t] = rssm_state.h.clone()
                    z_history[:, t] = rssm_state.z.clone()

            # Replica l'ultimo stato per allineare le dimensioni con il critic loss come nel tuo codice originale
            real_states = torch.cat([real_states, real_states[:, -1:]], dim=1)

            # === 3. FLATTENING (LA MAGIA DI DREAMER) ===
            # Fonde Batch e Tempo: B * T diventerà il nostro nuovo mega-batch per l'immaginazione
            BT = B * T
            
            s_flat = real_states[:, :-1].reshape(BT, self.emb_dim) # Escludiamo l'ultimo fittizio
            
            rssm_state_flat = RSSMState(
                h=h_history.reshape(BT, self.rssm.h_dim),
                z=z_history.reshape(BT, self.rssm.L, self.rssm.K)
            )

            if valid_actions_mask is not None:
                if valid_actions_mask.dim() == 3: # Se ha forma [B, T, action_dim]
                    current_action_mask = valid_actions_mask.reshape(BT, -1)
                else: # Se ha forma [B, action_dim]
                    current_action_mask = valid_actions_mask.repeat_interleave(T, dim=0)
            else:
                current_action_mask = None

            # Tensori per i risultati dell'immaginazione (Forma: [B*T, Horizon, ...])
            reward        = torch.empty(BT, self.horizon, device=device)
            continues     = torch.empty(BT, self.horizon, device=device)
            entropys      = torch.empty(BT, self.horizon, device=device)
            log_probs     = torch.empty(BT, self.horizon, device=device)
            latent_states = torch.empty(BT, self.horizon, self.emb_dim, device=device)

            # === 4. IMMAGINAZIONE TOTALE PARALLELA ===
            s = s_flat
            rssm_state = rssm_state_flat

            for t in range(self.horizon):
                act_t1, log_p, entropy = self.actor(s, current_action_mask)

                latent_states[:, t] = s

                r_t_real = self.dreamer_model.predict_reward(s)
                c_t      = self.dreamer_model.predict_continue(s)
                user_att_t_real = self.dreamer_model.predict_user_att(s)

                reward[:, t]    = r_t_real
                continues[:, t] = c_t
                log_probs[:, t] = log_p
                entropys[:, t]  = entropy

                act_t1_enc = self.dreamer_model.encode_action(act_t1)
                att_t1_enc = self.dreamer_model.encode_att(user_att_t_real)

                rssm_state, _ = self.rssm.imagine_step(rssm_state, act_t1_enc, att_t1_enc)
                
                z_flat_imag = rssm_state.z.flatten(1)
                s = self.dreamer_model.encode_state(rssm_state.h, z_flat_imag)
                
                # La maschera valida delle azioni ha senso solo al primo frame, poi l'agente è libero
                current_action_mask = None

            latent_states = torch.cat([latent_states, s.unsqueeze(1)], dim=1)

            return {
                "rewards": reward,           # [B*T, horizon]
                "continues": continues,      # [B*T, horizon]
                "entropy": entropys,         # [B*T, horizon]
                "imagination": latent_states,# [B*T, horizon+1, emb_dim]
                "log_probs": log_probs,      # [B*T, horizon]
                "real_states": real_states   # [B, T+1, emb_dim]
            }

    # 🎯 Metodo aggiornato: accetta init_action per copiare il comportamento di Dreamer.py
    def _apply_episode_reset(self, state: RSSMState, last_action: torch.Tensor, init_action: torch.Tensor, is_first: torch.Tensor) -> tuple[RSSMState, torch.Tensor]:
        if is_first is None or not is_first.any():
            return state, last_action

        fresh = self.rssm.init_state(state.h.shape[0], device=state.h.device)

        with torch.no_grad():
            first  = is_first.float().unsqueeze(-1)          
            first3 = first.unsqueeze(-1)                     

        state.h        = torch.where(first.bool(),  fresh.h, state.h)
        state.z        = torch.where(first3.bool(), fresh.z, state.z)
        # 🎯 Usa l'azione al passo t=0 (init_action) al posto degli zeri assoluti
        last_action    = torch.where(first.bool(), init_action, last_action)

        return state, last_action