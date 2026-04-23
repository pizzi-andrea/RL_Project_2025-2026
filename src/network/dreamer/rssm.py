# rssm_lightning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import *

# Custom Grucell multiblock
import torch
import torch.nn as nn

class DreamerGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks=8):
        super().__init__()

        assert hidden_size % num_blocks == 0

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks

        self.input_to_gates = nn.Linear(input_size, 3 * hidden_size, bias=False)

        self.recurrent_weight = nn.Parameter(
            torch.empty(num_blocks, self.block_size, 3 * self.block_size)
        )

        self.norm_r = nn.RMSNorm(hidden_size)
        self.norm_z = nn.RMSNorm(hidden_size)
        self.norm_n = nn.RMSNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_blocks):
            nn.init.xavier_uniform_(self.recurrent_weight[i])

    def forward(self, x, h_prev):
        B = x.shape[0]

        # Input
        gates_x = self.input_to_gates(x)  # [B, 3H]
        x_r, x_z, x_n = torch.chunk(gates_x, 3, dim=-1)

        # Recurrent (block-diagonal)
        h_blocks = h_prev.view(B, self.num_blocks, self.block_size).transpose(0, 1)
        gates_h = torch.bmm(h_blocks, self.recurrent_weight)  # [num_blocks, B, 3*block]

        
        h_r, h_z, h_n = torch.chunk(gates_h, 3, dim=2)

        h_r = h_r.transpose(0, 1).reshape(B, self.hidden_size)
        h_z = h_z.transpose(0, 1).reshape(B, self.hidden_size)
        h_n = h_n.transpose(0, 1).reshape(B, self.hidden_size)

        # GRU
        r = torch.sigmoid(self.norm_r(x_r + h_r))
        z = torch.sigmoid(self.norm_z(x_z + h_z))
        n = torch.tanh(self.norm_n(x_n + r * h_n))

        h = (1 - z) * n + z * h_prev
        return h

# -------------------------
# RSSM (discrete latent)
# -------------------------

@dataclass
class RSSMState:
    h: torch.Tensor      # [B,H]
    z: torch.Tensor  # [B, L, K] (posterior or prior logits)
   

class RSSM(nn.Module): # need specific train loop
    """
    Dreamer-style RSSM:
      h_t = GRU(h_{t-1}, [embed(z_{t-1}), embed(a_{t-1})])
      prior:    p(z_t | h_t)
      posterior q(z_t | h_t, enc(x_t))
    """
    def __init__( #x_t has shape [B,L,K]
        
        self,
        action_dim: int=16,
        h_dim: int = 1024, # deterministic hidden state dimension
        obs_dim:int = 1024,
        stoch_latents: int = 32,   # L (number of discrete latent variables)
        stoch_classes: int = 32,   # K (number of classes per latent variable)
        att_dim: int = 6,
        hidden_dim: int = 1024,
        num_gru_blocks: int = 8,
        unimix: float = 0.01,
    ):
        """
        Recurrent State Space Model (RSSM) that manages deterministic and stochastic state transitions.
        """
        super().__init__()
        # num gru_cell blocks
        self.num_gru_blocks = num_gru_blocks
        # action dim (num of actions)
        self.action_dim = action_dim
        
        # dimension of detterministic state (integer)
        self.h_dim = h_dim
        
        # one-hot encoding dimensions (num classes- num latent for class)
        self.L = stoch_latents
        self.K = stoch_classes
        
        # unimix value
        self.unimix = unimix
        
        # dimension of internal projection (integer)
        self.hidden_dim = hidden_dim
        
        # attribute game (integer)
        self.att_dim = att_dim
        

        self.z_embed = nn.Sequential(
            nn.Linear(self.K * self.L, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU()
        )

        self.z_dim = hidden_dim
        self.block_size = h_dim // self.num_gru_blocks
        self.gru_input_size = action_dim + att_dim + self.z_dim # z_emb + a_emb + h_mixed_proj
        
        print("expected GRU", self.gru_input_size)
        self.gru_cell = DreamerGRUCell(
            input_size=self.gru_input_size, 
            hidden_size=self.h_dim,  
            num_blocks=self.num_gru_blocks
        )
        
        # predict encoded rappresentation z_hat given GRUcell hidden state and input rappresentation
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim), 
            nn.RMSNorm(hidden_dim),          
            nn.SiLU(),
            nn.Linear(hidden_dim, self.K * self.L) 
        )

        # predict encoded rappresentation z_hat given only GRUcell hidden state 
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.K * self.L)
        )

    def init_state(self, batch_size: int, device=None) -> RSSMState:
        """Inizialize RSSM memory and return the state"""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.L, self.K, device=device)
      
       
        return RSSMState(h=h, z=z)

    def compute_stochastic(self, z):
    
        z_flat = torch.flatten(z, start_dim=1) # [B, 1024]
        z_out = self.z_embed(z_flat)

        return z_out 
    def _gru_step(self, h_t: torch.Tensor, z_t: torch.Tensor, a_t:torch.Tensor = None) -> torch.Tensor:
        
        
        z_flat  = self.compute_stochastic(z_t)

        if a_t is not None:
            z_flat = torch.cat([z_flat, a_t], dim=-1)

        
        h_t1 = self.gru_cell(z_flat, h_t)

        

        return h_t1
    
    def _apply_mask(self, state_old, h_new, z_new, mask=None):

        if mask is not None:
            m  = mask.bool().unsqueeze(-1)   # [B,1]
            m3 = mask.bool().view(-1,1,1)    # [B,1,1]

            h = torch.where(m, h_new, state_old.h)

            # structured state ONLY
            z_struct = torch.where(m3, z_new, state_old.z)


        else:
            h = h_new
            z_struct = z_new
       
        return RSSMState(h=h, z=z_struct)

   

    def observe_step(
        self,
        state: RSSMState,
        action_t: torch.Tensor,
        x_t1: torch.Tensor,
        att_t: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, torch.Tensor]:

        a_t = torch.cat([action_t, att_t], dim=-1)
        
        z_t = state.z
        h_t = state.h

        
       
        h_t1, z_hat_t1_log_prior = self._compute_prior(h_t, z_t, a_t)
        z_hat_t1_log_posterior   = self._compute_posterior(h_t1, x_t1)

        z_t1               = logits2discrate(z_hat_t1_log_posterior)
        
        #print(z_hat_t1_log_prior.shape)
        #print(z_hat_t1_log_posterior.shape)
        
        return self._apply_mask(state, h_t1, z_t1, mask_t), z_hat_t1_log_prior, z_hat_t1_log_posterior
                            


    def imagine_step(
        self,
        state: RSSMState,
        action: torch.Tensor,
        user_att:torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, torch.Tensor]:

        a_t = torch.cat([action, user_att], dim=-1)

        # use flat version of z
        z_t = state.z
        
        # current memory state
        h_t = state.h

        # memory + action = new state s t+1
        h_t1, z_hat_t1_log = self._compute_prior(h_t, z_t, a_t)

        # encode raw z logits in one-hot categorical distribution
        z_hat_t1           = logits2discrate(z_hat_t1_log, self.unimix)

        return self._apply_mask(state, h_t1, z_hat_t1, mask=mask_t), z_hat_t1_log
                        

    def _compute_prior(self, h_t, z_t, a_t): # return h_t1, z_hat_t1_log
        
        B = z_t.shape[0]
        # hidden state (t+1)
        h_t1 = self._gru_step(h_t, z_t, a_t)
        
        # logits for stochastic state
        z_hat_t1_log = self.prior_net(h_t1) # use only memory (prior knowledge)
        z_hat_t1_log = z_hat_t1_log.reshape(B, self.L, self.K)
        

        return h_t1, z_hat_t1_log

    def _compute_posterior(self, h_t1, x_t1): 
        B = h_t1.shape[0]
        # use memory + prior knowledge (obs, att etc...)
        posterior = torch.cat([h_t1, x_t1], dim=-1)
        # logits for stochastic state
        z_hat_t1_log = self.post_net(posterior)
        z_hat_t1_log = z_hat_t1_log.reshape(B, self.L, self.K)
        
        return z_hat_t1_log

    
