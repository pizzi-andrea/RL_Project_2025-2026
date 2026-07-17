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
        
        
        # == layer l1
        self.l1 = nn.Sequential(
            nn.Linear(in_features=dim_state, out_features=hidden_dim*2),
            nn.RMSNorm(hidden_dim*2),
            nn.SiLU(),
        )

        # == layer l2
        self.l2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
        )

        # == layer l3
        self.l3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
        )

        # == output layers
        self.dist_pred = nn.Linear(in_features=hidden_dim, out_features=num_dist)
        
        # == low variance init (improve convergence rate)
        torch.nn.init.orthogonal_(self.dist_pred.weight, gain=1.0) 
        self.dist_pred.bias.data.fill_(0.0)
    
  
    def forward(self, x, valid_actions_mask=None):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        
        logits = self.dist_pred(x)

        # mask invalid actions
        if valid_actions_mask is not None:
            mask = valid_actions_mask.bool()
            logits = logits.masked_fill(~mask, -1e9)

        # log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # unimix that prevent actions collapse (zero probs)
        if valid_actions_mask is not None:
            num_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = mask.float() / num_valid
        else:
            uniform = torch.full_like(logits, 1.0 / self.num_dist)
            
        mixed_probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        
        
        mixed_log_probs = torch.log(mixed_probs + 1e-8)
        
        # eliminate the entropy for invalid actions
        if valid_actions_mask is not None:
            mixed_log_probs = mixed_log_probs.masked_fill(~mask, -1e9)

        # init dist
        dist = OneHotCategorical(logits=mixed_log_probs)

        # use different sampling strategy that depend by train phase
        
        action = dist.sample() 
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    



    

