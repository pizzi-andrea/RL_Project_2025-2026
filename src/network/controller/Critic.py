from torch import  nn
from .utils import *
import torch

class Critic(nn.Module):
    """Network for Critic rule implemented with support at Robust predictor method"""
    def __init__(self, dim_state, num_buckets, hidden_dim=1024):
        super().__init__()
        self.num_buckets = num_buckets
        self.register_buffer("buckets", symexp( torch.linspace(-20, 20, num_buckets) ) )
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
        self.softmax = nn.Softmax()

    
  

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        pred_values_log =  self.buckets_proj(x)
        pred_values_prob = self.softmax(pred_values_log)

        # -- weighted mean
        pred_values = (pred_values_prob * self.buckets).sum(dim=-1)

        return pred_values_log, pred_values
        

     