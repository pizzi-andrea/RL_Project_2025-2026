import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import OneHotCategorical
from .utils import *
class Actor(nn.Module):
    """Actor network for DreamerV3"""
    def __init__(self, dim_state:int, num_dist:int, hidden_dim:int=1024, unimix:float=0.01):
        """
        Configure Actor with homogenues action space and specific space dimension

        Args:
            dim_state(int): dimension of state space
            num_dist(int): number of actions
            hiden_dim(int): dimension for mlp layers
            unimix(float): level of unimix set to prevent action space collapse 
        """
        super().__init__()
        self.dim_state = dim_state
        self.num_dist = num_dist
        self.unimix = unimix
        
        
        # == mlp layers
        self.l1 = nn.Sequential(
            nn.Linear(in_features=dim_state, out_features=hidden_dim*2, bias=False),
            nn.RMSNorm(hidden_dim*2, eps=0.001),
            nn.SiLU(),
        )

        self.l2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim, eps=0.001),
            nn.SiLU(),
        )

        self.l3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim, eps=0.001),
            nn.SiLU(),
        )
        # ================================================================================

        # == output layers
        self.dist_pred = nn.Linear(in_features=hidden_dim, out_features=num_dist, bias=True)
        
        # == low variance init (improve convergence rate)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(self, 'dist_pred') and m is self.dist_pred:
             
                nn.init.uniform_(m.weight, -1e-4, 1e-4) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                dreamerv3_normal_initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    
    def forward(self, x, valid_actions_mask=None): 
        """ 
        decode world model state x in action then return
        choosen action, log_probs and entropy

        Args:
            x(torch.Tensor): world model state
            valid_actions_mask(torch.Tensor): binary mask for valid actions if avaiable
        
        """
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        
        logits = self.dist_pred(x)

        # mask invalid actions
        if valid_actions_mask is not None:
            mask = valid_actions_mask.bool()
            logits = logits.masked_fill(~mask, -1e9)

        # == log_softmax
        probs = torch.softmax(logits, dim=-1)

        # == unimix that prevent actions collapse (zero probs)
        if valid_actions_mask is not None:
            num_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = mask.float() / num_valid
        else:
            uniform = torch.full_like(logits, 1.0 / self.num_dist)
            
        mixed_probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        
        # == init dist =========================================
        dist = OneHotCategorical(probs=mixed_probs)
        
        # == STE =============================================
        sampled_action = dist.sample()
        action = sampled_action.float().detach() + (mixed_probs - mixed_probs.detach())

        # == use different sampling strategy that depend by train phase
        
        log_prob = dist.log_prob(sampled_action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    



    

