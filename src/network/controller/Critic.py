from torch import  nn
from .utils import *
import torch

class Critic(nn.Module):
    """Network for Critic rule implemented with support at Robust predictor method"""
    def __init__(self, dim_state, num_buckets, min_buckets=-20, max_buckets=20, hidden_dim=1024):
        super().__init__()

        
        self.num_buckets = num_buckets
        
        self.min_buckets = min_buckets
        self.max_buckets = max_buckets

        self.register_buffer("buckets", torch.linspace(min_buckets, max_buckets, num_buckets)  ) 
        

        self.net = self._build_mlp(dim_state, hidden_dim)
        self.buckets_proj = nn.Linear(hidden_dim, num_buckets, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)
   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            dreamerv3_normal_initializer(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
   
    def _build_mlp(self, dim_state, hidden_dim):
        """Helper per costruire MLP senza bias nei layer nascosti e con LayerNorm"""
        return nn.Sequential(
            nn.Linear(dim_state, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2, eps=0.001),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=0.001),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=0.001),
            nn.SiLU(),
        )
    def forward(self, x):
        x = self.net(x)

        pred_values_log =  self.buckets_proj(x)
        pred_values_prob = self.softmax(pred_values_log)

        
        pred = symexp ( (pred_values_prob * self.buckets).sum(dim=-1) )

        # return both values in symlog and symexp(linear)
        return pred_values_log, pred
        

     