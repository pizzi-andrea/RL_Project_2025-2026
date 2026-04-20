import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import OneHotCategorical

class Actor(nn.Module):
    """Actor network with unimix"""
    def __init__(self, dim_state, num_dist, hidden_dim=1024, unimix=0.01):
        super().__init__()
        self.dim_state = dim_state
        self.num_dist = num_dist
        self.unimix = unimix
        
        
        # -- actor network layers
        self.l1 = nn.Sequential(
            nn.Linear(in_features=dim_state, out_features=hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
        )

        self.dist_pred = nn.Linear(in_features=hidden_dim, out_features=num_dist)
        
        # -- inizialize weights for low variance
        torch.nn.init.orthogonal_(self.dist_pred.weight, gain=1.0) 
        self.dist_pred.bias.data.fill_(0.0)
    
  
    def forward(self, x, valid_actions_mask=None):
        x = self.l1(x)
        x = self.l2(x)
        logits = self.dist_pred(x)

        # --- MASK ---
        if valid_actions_mask is not None:
            mask = valid_actions_mask.bool()
            logits = logits.masked_fill(~mask, -1e9)

            # uniform SOLO sulle azioni valide
            num_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = mask.float() / num_valid
        else:
            uniform = torch.full_like(logits, 1.0 / self.num_dist)

        # --- LOG-SOFTMAX (STABILE) ---
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # --- UNIMIX (DreamerV3 style) ---
        probs = (1.0 - self.unimix) * probs + self.unimix * uniform

        # convert in log-prob
        log_probs = torch.log(probs + 1e-8)

        dist = OneHotCategorical(probs=probs)

        # --- ACTION ---
        if self.training:
            action = dist.sample()
        else:
            index = probs.argmax(dim=-1, keepdim=True)
            action = torch.zeros_like(probs).scatter_(-1, index, 1.0)

        log_prob = dist.log_prob(action)
        entropy =dist.entropy()

        return action, log_prob, entropy

    



    

