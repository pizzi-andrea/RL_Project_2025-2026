# rssm_lightning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import unimix_probs, st_onehot_from_logits



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
        stoch_latents: int = 32,   # L (number of discrete latent variables)
        stoch_classes: int = 32,   # K (number of classes per latent variable)
        embed_dim: int = 1024,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.L = stoch_latents
        self.K = stoch_classes
        self.unimix = unimix

        self.action_embed = nn.Sequential(nn.Linear(action_dim, embed_dim), nn.SiLU())
        self.z_embed = nn.Sequential(nn.Linear(self.L * self.K, embed_dim), nn.SiLU()) #so that action and z have same dimension that is not too smal

        self.gru = nn.GRU(input_size=2 * embed_dim, hidden_size=deter_dim, batch_first=True) #Batch, seq, feature
        self.loss_buffer = None
        # prior logits from h
        self.prior_net = nn.Sequential( # self.prior_net(h) -> logits of z

            nn.Linear(deter_dim, deter_dim),
            nn.SiLU(),
            nn.Linear(deter_dim, self.L * self.K),
        )
        # posterior logits from [h, enc(x)]
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, deter_dim),
            nn.SiLU(),
            nn.Linear(deter_dim, self.L * self.K),
        )

    def init_state(self, batch_size: int, device=None) -> RSSMState:
        """Inizialize RSSM memory and return the state"""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.L, self.K, device=device)
        z = torch.zeros(batch_size, self.L * self.K, device=device)
        return RSSMState(h=h, z=z, z_logits=z_logits)

    def _logits_to_st_z(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: #conversion logit-> discrete z
        # logits: [B,L*K] -> [B,L,K]
        logits = logits.view(-1, self.L, self.K)
        probs = torch.softmax(logits, dim=-1) #softmax to obtain a probability
        probs = unimix_probs(probs, unimix=self.unimix) #unimix that adds some uniform noise
        # incorporate unimix back into logits (stable)
        logits = torch.log(probs.clamp_min(1e-8)) # stable logits after unimix
        z_st = st_onehot_from_logits(logits)  # [B,L,K] z discrete but derivable
        z_flat = z_st.view(-1, self.L * self.K)
        return z_flat, logits
    
    def _gru_step(self, prev_h: torch.Tensor, inp_bt: torch.Tensor) -> torch.Tensor:
        # prev_h: [B,H]
        # inp_bt: [B, 2*embed_dim]
        x = inp_bt.unsqueeze(1)  # [B,1,2*embed_dim] for GRU
        h0 = prev_h.unsqueeze(0)  # [1,B,H] for GRU
        _, h_last = self.gru(x, h0)  # h_last: [1,B,H]
        return h_last.squeeze(0)  # [B,H]
    
    def imagine_step(self, prev: RSSMState, action: torch.Tensor, mask_t: Optional[torch.Tensor]= None) -> RSSMState: # [B] 1=valid 0=pad
        """Predict next state using the 'immagination' mechanism, no input obs are required and return network state"""
        aemb = self.action_embed(action)
        zemb = self.z_embed(prev.z)
        inp = torch.cat([aemb, zemb], dim=-1)

        h_new = self._gru_step(prev.h, inp)

        prior_logits = self.prior_net(h_new)
        z_new, prior_logits = self._logits_to_st_z(prior_logits)

        if mask_t is not None:
            m = mask_t.float().unsqueeze(-1)  # [B,1]
            h = m * h_new + (1 - m) * prev.h
            z = m * z_new + (1 - m) * prev.z
            m3 = mask_t.float().view(-1, 1, 1)
            z_logits = m3 * prior_logits + (1 - m3) * prev.z_logits
            return RSSMState(h=h, z=z, z_logits=z_logits)

        return RSSMState(h=h_new, z=z_new, z_logits=prior_logits)

    def observe_step(
        self,
        prev: RSSMState,
        action: torch.Tensor,
        enc_x: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, torch.Tensor]:
        """Predict next state from observation and return network state"""
        aemb = self.action_embed(action)
        zemb = self.z_embed(prev.z)
        inp = torch.cat([aemb, zemb], dim=-1)

        h_new = self._gru_step(prev.h, inp)

        prior_logits = self.prior_net(h_new).view(-1, self.L, self.K)

        post_in = torch.cat([h_new, enc_x], dim=-1)
        post_logits = self.post_net(post_in).view(-1, self.L, self.K)

        post_probs = unimix_probs(torch.softmax(post_logits, dim=-1), unimix=self.unimix)
        post_logits = torch.log(post_probs.clamp_min(1e-8))
        z_st = st_onehot_from_logits(post_logits)
        z_new = z_st.view(-1, self.L * self.K)

        if mask_t is not None:
            m = mask_t.float().unsqueeze(-1)
            h = m * h_new + (1 - m) * prev.h
            z = m * z_new + (1 - m) * prev.z
            m3 = mask_t.float().view(-1, 1, 1)
            z_logits = m3 * post_logits + (1 - m3) * prev.z_logits
            state = RSSMState(h=h, z=z, z_logits=z_logits)
            return state, prior_logits

        state = RSSMState(h=h_new, z=z_new, z_logits=post_logits)
        return state, prior_logits 


