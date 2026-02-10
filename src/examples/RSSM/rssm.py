# rssm_lightning.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from .utils import *

import torch
import torch.nn as nn

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
        action_dim: int,
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

        self.gru = nn.GRUCell(input_size=2 * embed_dim, hidden_size=deter_dim) #input: [B,input_size], prev_h: [B,deter_dim] -> out: [B,deter_dim]
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
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z_logits = torch.zeros(batch_size, self.L, self.K, device=device)
        z = torch.zeros(batch_size, self.L * self.K, device=device)
        return RSSMState(h=h, z=z, z_logits=z_logits)

    

    def imagine_step(self, prev: RSSMState, action: torch.Tensor) -> RSSMState:
        # action: [B,A] one-hot or continuous vector
        aemb = self.action_embed(action)
        zemb = self.z_embed(prev.z)
        inp = torch.cat([aemb, zemb], dim=-1)
        h = self.gru(inp, prev.h)

        prior_logits = self.prior_net(h)
        z, prior_logits =  logits_to (prior_logits)
        return RSSMState(h=h, z=z, z_logits=prior_logits)

    def observe_step(self, prev: RSSMState, action: torch.Tensor, enc_x: torch.Tensor) -> Tuple[RSSMState, torch.Tensor]: #it's like imagine step but with posterior (enc_x is the encoded observation)
        # First update h with action and prev z, then compute posterior using enc_x
        aemb = self.action_embed(action)
        zemb = self.z_embed(prev.z)
        inp = torch.cat([aemb, zemb], dim=-1)
        h = self.gru(inp, prev.h)

        prior_logits = self.prior_net(h).view(-1, self.L, self.K)

        post_in = torch.cat([h, enc_x], dim=-1)
        post_logits = self.post_net(post_in).view(-1, self.L, self.K)

        # sample z from posterior (straight-through)
        post_probs = unimix_probs(torch.softmax(post_logits, dim=-1), unimix=self.unimix)
        post_logits = torch.log(post_probs.clamp_min(1e-8))
        z_st = st_onehot_from_logits(post_logits)
        z_flat = z_st.view(-1, self.L * self.K)

        state = RSSMState(h=h, z=z_flat, z_logits=post_logits)
        # return prior_logits for KLs (that is needed for the training loss and needs prior and posterior)
        return state, prior_logits 
