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
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        
        # 1. Proiezione dell'input verso tutti i gate di tutti i blocchi
        # DreamerV3 usa una proiezione densa per l'input (zt, at, h_prev_mixed)
        self.input_to_gates = nn.Linear(input_size, 3 * hidden_size, bias=False)
        
        # 2. Matrice Ricorrente Block-Diagonal (Implementata come Linear "Grouped")
        # Usiamo un tensore [num_blocks, block_size, 3 * block_size] per calcolare i gate ricorrenti in parallelo
        self.recurrent_weight = nn.Parameter(torch.empty(num_blocks, self.block_size, 3 * self.block_size))
        self.recurrent_bias = nn.Parameter(torch.empty(3 * hidden_size))
        
        # 3. RMSNorm per la stabilizzazione (Critica in DreamerV3)
        self.layer_norm = nn.RMSNorm(3 * hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.recurrent_weight)
        nn.init.zeros_(self.recurrent_bias)

    def forward(self, x, h_prev):
        # x: [B, input_size]
        # h_prev: [B, hidden_size]
        B = x.shape[0]
        

        # Calcolo gate dall'input (Densa)
        gates_x = self.input_to_gates(x) # [B, 3 * hidden_size]
        x = self.layer_norm(gates_x)
        #gates_x = self.layer_norm(x)

        # Calcolo gate ricorrenti (Block-Diagonal)
        # Reshape h_prev per operazione a blocchi: [B, num_blocks, block_size]
        h_reshape = h_prev.view(B, self.num_blocks, self.block_size).transpose(0, 1) # [num_blocks, B, block_size]
        
        # Batch Matrix Multiplication parallela sui blocchi
        gates_h = torch.bmm(h_reshape, self.recurrent_weight) # [num_blocks, B, 3 * block_size]
        gates_h = gates_h.transpose(0, 1).reshape(B, 3 * self.hidden_size) # [B, 3 * hidden_size]

        # Somma gate e bias
        gates = gates_x + gates_h + self.recurrent_bias
        
        # Split gates: Reset (r), Update (z), New (n)
        r, z, n = torch.split(gates, self.hidden_size, dim=-1)
        
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n * r) # Reset gate applicato prima del tanh
        
        h_new = (1 - z) * n + z * h_prev
        
        
        return h_new

# -------------------------
# RSSM (discrete latent)
# -------------------------





@dataclass
class RSSMState:
    h: torch.Tensor      # [B,H]
    z: torch.Tensor      # [B, L*K] (flattened one-hot blocks)
    z_logits: torch.Tensor  # [B, L, K] (posterior or prior logits)

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
        deter_dim: int = 1024, # deterministic hidden state dimension
        obs_dim:int = 1024,
        stoch_latents: int = 32,   # L (number of discrete latent variables)
        stoch_classes: int = 32,   # K (number of classes per latent variable)
        embed_dim: int = 1024,
        num_gru_blocks: int = 8,
        unimix: float = 0.01,
    ):
        """
        Recurrent State Space Model (RSSM) that manages deterministic and stochastic state transitions.
        
        :param action_dim: Dimension of the action space.
        :type action_dim: int
        :param deter_dim: Dimension of the GRU's deterministic hidden state (h).
        :type deter_dim: int
        :param stoch_latents: Number of discrete latent variables (L) in the stochastic state (z).
        :type stoch_latents: int
        :param stoch_classes: Number of categories (K) for each discrete latent variable.
        :type stoch_classes: int
        :param embed_dim: Internal projection dimension for actions, states, and observations.
        :type embed_dim: int
        :param unimix: Mixing ratio for uniform distribution in the categorical prior to prevent collapse.
        :type unimix: float
        """
        super().__init__()
        self.num_gru_blocks = num_gru_blocks
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.L = stoch_latents
        self.K = stoch_classes
        self.unimix = unimix
        self.output_dimension = self.L*self.K
        #self.gru_input_proj_dim = gru_projection_dim

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU()
        )
        self.z_embed = nn.Sequential(
            nn.Linear(self.output_dimension, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU()
        )

        """self.gru_input_proj = nn.Sequential(
            nn.Linear(2*embed_dim, gru_projection_dim),
            nn.RMSNorm(gru_projection_dim), # 
            nn.SiLU()
        )"""

        #self.gru = nn.GRUCell(input_size=2*embed_dim, hidden_size=deter_dim) #Batch, seq, feature

        self.h_proj = nn.Sequential(
            nn.Linear(deter_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU()
        )
        self.block_size = deter_dim // self.num_gru_blocks
        
        gru_input_size = 3 * embed_dim # z_emb + a_emb + h_mixed_proj
        self.gru_cell = DreamerGRUCell(
            input_size=gru_input_size, 
            hidden_size=self.deter_dim,  
            num_blocks=self.num_gru_blocks
        )
        
        # predict encoded rappresentation z_hat given GRUcell hidden state and input rappresentation
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, embed_dim), 
            nn.RMSNorm(embed_dim),          
            nn.SiLU(),
            nn.Linear(embed_dim, self.output_dimension) 
        )

        # predict encoded rappresentation z_hat given only GRUcell hidden state 
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, self.output_dimension)
        )

    def init_state(self, batch_size: int, device=None) -> RSSMState:
        """Inizialize RSSM memory and return the state"""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.L, self.K, device=device)
        z = torch.zeros(batch_size, self.L * self.K, device=device)
        return RSSMState(h=h, z=z, z_logits=z_logits)

    
    def _gru_step(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        
        h_mixed = self.h_proj(h_t) 
        # Unione di latente, azione e mixing (3 * embed_dim = 768)
        gru_input = torch.cat([z_t, h_mixed], dim=-1) 
        h_t1 = self.gru_cell(gru_input, h_t)

        return h_t1
    
    
    
    
    def _apply_mask(self, h_new, z_new, prior_logits, post_logits=None, mask=None):
        """
        Applica il masking allo stato RSSM e seleziona i logit corretti (Posterior se presente, altrimenti Prior).
        """
        # Se post_logits è None (es. in fase di immaginazione), usiamo il prior come riferimento per lo stato
        effective_logits = post_logits if post_logits is not None else prior_logits

        if mask is not None:
            # Espansione del mask per il broadcasting con h [B, H] e z [B, L*K]
            m = mask.float().unsqueeze(-1)
            # Espansione per i logit [B, L, K]
            m3 = mask.float().view(-1, 1, 1)
            
            h = h_new * m
            z = z_new * m
            z_logits = effective_logits * m3
            
            state = RSSMState(h=h, z=z, z_logits=z_logits)
        else:
            state = RSSMState(h=h_new, z=z_new, z_logits=effective_logits)
        
        return state, prior_logits

    def _compute_prior(self, h_t, z_t, a_t): # return h_t1, z_hat_t1_log
        #x = self.gru_input_proj(x)
        seq_t = torch.cat([z_t, a_t], dim=-1) # 2* embed_dim
        h_t1 = self._gru_step(h_t, seq_t)
        z_hat_t1_log = self.prior_net(h_t1).view(-1, self.L, self.K)

        return h_t1, z_hat_t1_log

    def _compute_posterior(self, h_t1, x_t1): # return z_t_log
        posterior = torch.cat([h_t1, x_t1], dim=-1) # add observation
        z_t1_log = self.post_net(posterior).view(-1, self.L, self.K)
        
        return z_t1_log

    def imagine_step(self, state: RSSMState, action: torch.Tensor, mask_t: Optional[torch.Tensor]= None) -> Tuple[RSSMState, torch.Tensor]:
        """Predict next state using the 'imagination' mechanism."""
        
        a_t = self.action_embed(action)
        z_t = self.z_embed(state.z)
        h_t = state.h
       
        h_t1, z_hat_t1_log = self._compute_prior(h_t, z_t, a_t)
        # new sample from dist z_hat_t1
        z_hat_t1 = logits2discrate(z_hat_t1_log, self.unimix)
        return self._apply_mask(h_t1, z_hat_t1, z_hat_t1_log, mask=mask_t)
    

    def observe_step(
        self,
        state: RSSMState,
        action_t: torch.Tensor,
        x_t1: torch.Tensor, # t+1 !
        mask_t: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, torch.Tensor]:
        """Predict next state from observation and return network state"""
        
        # -- convert in homogeneus latent spaces
        a_t = self.action_embed(action_t)   # a_t
        z_t = self.z_embed(state.z)         # z_t
        h_t = state.h                       # h_t
        # -- 
        
        #print(f"DEBUG: emb_action shape {emb_action.shape}, emb_latent shape {emb_latent.shape}")
        #exit()
        
        h_t1, z_hat_t1_log = self._compute_prior(h_t, z_t, a_t)
        z_t1_log = self._compute_posterior(h_t1, x_t1)
        
        # Unimix Dreamer V3
        z_t1 = logits2discrate(z_t1_log)

        return self._apply_mask(h_t1, z_t1, z_hat_t1_log, z_t1_log, mask_t)
