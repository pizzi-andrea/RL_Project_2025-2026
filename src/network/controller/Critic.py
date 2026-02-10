from torch import  nn
import torch

class Critic(nn.Module):
    def __init__(self, dim_state, num_buckets, hidden_dim=1024):
        super().__init__()
        self.num_buckets = num_buckets
        self.register_buffer("buckets", torch.linspace(-20, 20, num_buckets))
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
        
        self.return_predictor = nn.Linear(in_features=hidden_dim, out_features=num_buckets)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return self.return_predictor(x)

     