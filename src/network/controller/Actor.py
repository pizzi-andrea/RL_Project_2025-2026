import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Independent, OneHotCategorical

class Actor(nn.Module):
    def __init__(self, dim_state, num_dist, hidden_dim=1024, unimix=0.01, one_hot: bool = True):
        super().__init__()
        self.dim_state = dim_state
        self.num_dist = num_dist
        self.unimix = unimix
        self.one_hot = one_hot 

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
        
        torch.nn.init.orthogonal_(self.dist_pred.weight, gain=0.01) 
        self.dist_pred.bias.data.fill_(0.0)

    def reparameterize_categorical(self, logits):
        return F.gumbel_softmax(logits, tau=1.0, hard=True)
    
    def reparameterize_bernoulli(self, probs):
        sample = torch.bernoulli(probs)
        return probs + (sample - probs).detach()
    
    def forward(self, x, valid_actions_mask=None):
        x = self.l1(x)
        x = self.l2(x)

        logits = self.dist_pred(x)
        if valid_actions_mask is not None:
            logits = logits.masked_fill(~valid_actions_mask.bool(), -1e9)

        if self.one_hot:
            probs = F.softmax(logits, dim=-1)
            if valid_actions_mask is not None:
                uniform_noise = (self.unimix / valid_actions_mask.sum(dim=-1, keepdim=True)) * valid_actions_mask
                probs = (1.0 - self.unimix) * probs + uniform_noise
            else:
                probs = (1.0 - self.unimix) * probs + (self.unimix / self.num_dist)
                
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            logits_clean = torch.log(probs + 1e-8)
            action = self.reparameterize_categorical(logits_clean)
            dist = OneHotCategorical(logits=logits_clean)
        else:
            probs = torch.sigmoid(logits)
            probs = (1.0 - self.unimix) * probs + (self.unimix * 0.5)
            if valid_actions_mask is not None:
                probs = probs * valid_actions_mask
            action = self.reparameterize_bernoulli(probs)
            dist = Independent(Bernoulli(probs=probs), reinterpreted_batch_ndims=1)
        return action, dist

    



    

