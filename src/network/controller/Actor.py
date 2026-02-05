import torch
from torch import  nn
from torch.distributions import Bernoulli, Independent
class Actor(nn.Module):
    def __init__(self, dim_state, num_dist, hidden_dim=1024, unimix=0.01):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=dim_state, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
        )
        self.l2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LayerNorm(dim_state),
            nn.SiLU(),
        )
        
        self.num_dist = num_dist
        self.dist_pred = nn.Linear(in_features=hidden_dim, out_features=num_dist)
        

    # 
        self.dist_pred.weight.data.zero_()
        self.dist_pred.bias.data.zero_()
        self.unimix = unimix

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        
        logits = self.dist_pred(x)
        probs = torch.sigmoid(logits)
        unimix = self.unimix
        probs = (1.0 - unimix) * probs + (unimix * 0.5)
        dist = Independent(Bernoulli(probs=probs), reinterpreted_batch_ndims=1)
        
        return dist
        




