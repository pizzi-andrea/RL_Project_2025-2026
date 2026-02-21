# rssm_lightning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import *

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
        gru_input_size = 3 * embed_dim
        self.gru = nn.ModuleList(
            [nn.GRUCell(input_size=gru_input_size, hidden_size=self.block_size) for _ in range(self.num_gru_blocks)]
        )
        # prior logits memory 
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, embed_dim), 
            nn.LayerNorm(embed_dim),          
            nn.SiLU(),
            nn.Linear(embed_dim, self.output_dimension) 
        )
        # posterior logits observation + memory
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
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

    
    def _gru_step(self, prev_h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        
        h_proj_feat = self.h_proj(prev_h)

        gru_input = torch.cat([z, h_proj_feat], dim=-1)

        # split grup inputs in sublocks

        gru_input_blocks = torch.split(prev_h, self.block_size, dim=-1)
        h_next = []

        for i, cell in enumerate(self.gru):
            h_next.append( cell(gru_input, gru_input_blocks[i]) )

        output_embeds = torch.cat(h_next, dim=-1)

        return output_embeds
    
    
    def imagine_step(self, state: RSSMState, action: torch.Tensor, mask_t: Optional[torch.Tensor]= None) -> Tuple[RSSMState, torch.Tensor]:
        """Predict next state using the 'imagination' mechanism."""
        
        aemb = self.action_embed(action)
        zemb = self.z_embed(state.z)
        x = torch.cat([aemb, zemb], dim=-1)
        
        state_next_h, prior_logits_pure = self.compute_prior(state, x)        
        z_new_flat, _ = logits2discrate(prior_logits_pure, self.L, self.K, self.unimix)

        
        
        
        return self.apply_mask(state_next_h, z_new_flat, state, prior_logits_pure, prior_logits_pure, mask_t)
    
    def apply_mask(self, h_new, z_new, prev, post_logits, prior_logits, mask=None):
        """
        """
        if mask is not None:
            m = mask.float().unsqueeze(dim=-1)
            
           
            h = m * h_new 
            z = m * z_new
            
            m3 = mask.float().view(-1, 1, 1)
            
            z_logits = m3 * post_logits 
            
            #
            state = RSSMState(h=h, z=z, z_logits=z_logits)
        else:
            state = RSSMState(h=h_new, z=z_new, z_logits=post_logits)
        
        return state, prior_logits

    def compute_prior(self, hidden_state, x):
        #x = self.gru_input_proj(x)
        state_next = self._gru_step(hidden_state.h, x)
        prior_logits = self.prior_net(state_next).view(-1, self.L, self.K)

        return state_next, prior_logits

    def compute_posterior(self, state, observation):
        posterior = torch.cat([state, observation], dim=-1) # add observation
        posterior_logits = self.post_net(posterior).view(-1, self.L, self.K)
        
        return posterior_logits


    def observe_step(
        self,
        state: RSSMState,
        action: torch.Tensor,
        obs_encoded: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, torch.Tensor]:
        """Predict next state from observation and return network state"""
        
        # -- convert in homogeneus latent spaces
        emb_action = self.action_embed(action)
        emb_latent = self.z_embed(state.z)
        # -- 
        
        #print(f"DEBUG: emb_action shape {emb_action.shape}, emb_latent shape {emb_latent.shape}")
        #exit()
        x = torch.cat([emb_action, emb_latent], dim=-1)
        
        
        h, prior_logits = self.compute_prior(state, x)

        
        posterior_logits = self.compute_posterior(h, obs_encoded)
        # Unimix Dreamer V3
        
        posterior_probs = unimix_probs(torch.softmax(posterior_logits, dim=-1), unimix=self.unimix)
        posterior_logits_mix = torch.log_softmax(posterior_probs.clamp_min(1e-8), dim=-1)
        ## --
        
        z_new = st_onehot(posterior_logits_mix)
        z_new = z_new.view(-1, self.L * self.K)

        return self.apply_mask(
            h,  
            z_new, 
            state, 
            posterior_logits, 
            prior_logits, 
            mask_t
        )
        


