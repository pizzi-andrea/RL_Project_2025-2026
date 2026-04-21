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
        
        self.buckets_proj = nn.Linear(in_features=hidden_dim, out_features=num_buckets)
        self.softmax = nn.Softmax(-1)

        nn.init.xavier_uniform_(self.buckets_proj.weight)
        nn.init.zeros_(self.buckets_proj.bias) 
   
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        pred_values_log =  self.buckets_proj(x)
        pred_values_prob = self.softmax(pred_values_log)

        # -- weighted mean
        pred = symexp ( (pred_values_prob * self.buckets).sum(dim=-1) )
    
        return pred_values_log, pred
        

     