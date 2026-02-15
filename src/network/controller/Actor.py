import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Independent, OneHotCategorical

class Actor(nn.Module):
    """Actor network with unimix"""
    def __init__(self, dim_state, num_dist, hidden_dim=1024, unimix=0.01, one_hot: bool = True):
        super().__init__()
        self.dim_state = dim_state
        self.num_dist = num_dist
        self.unimix = unimix
        self.one_hot = one_hot 
        
        # -- neteork layers
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
        torch.nn.init.orthogonal_(self.dist_pred.weight, gain=0.01) 
        self.dist_pred.bias.data.fill_(0.0)

    #def reparameterize_categorical(self, logits):
    #    return F.gumbel_softmax(logits, tau=1.0, hard=True)
    
    #def reparameterize_bernoulli(self, probs):
    #    sample = torch.bernoulli(probs)
    #    return probs + (sample - probs).detach()
    
    # In Actor.py
    
    def forward(self, x, valid_actions_mask=None):
        x = self.l1(x)
        x = self.l2(x)

        logits = self.dist_pred(x)
        
        # --- Masking delle azioni (Mantiene la tua logica corretta) ---
        if valid_actions_mask is not None:
            mask = valid_actions_mask.bool()
            logits = logits.masked_fill(~mask, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            # Unimix (Paper Eq. 10: mix con uniforme per esplorazione)
            num_valid = mask.sum(dim=-1, keepdim=True)
            uniform = mask.float() / num_valid
        else:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.full_like(probs, 1.0 / self.num_dist)

        # Applicazione Unimix (Eq. 10)
        probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        
       
        dist = OneHotCategorical(probs=probs)
        
        # Campionamento STOCASTICO (Senza gradienti sull'azione!)
        # Questo è fondamentale per REINFORCE. L'azione è un "numero", non un tensore differenziabile.
        action = dist.sample() 
        
        return action, dist

    



    

