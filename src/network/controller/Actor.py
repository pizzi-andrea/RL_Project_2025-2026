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
        torch.nn.init.orthogonal_(self.dist_pred.weight, gain=1.0) 
        self.dist_pred.bias.data.fill_(0.0)

    def reparameterize_categorical(self, logits):
        return F.gumbel_softmax(logits, tau=1.0, hard=True)
    
    #def reparameterize_bernoulli(self, probs):
    #    sample = torch.bernoulli(probs)
    #    return probs + (sample - probs).detach()
    
    # In Actor.py
    def forward(self, x, valid_actions_mask=None):
        x = self.l1(x)
        x = self.l2(x)
        logits = self.dist_pred(x)
        
        if valid_actions_mask is not None:
            mask = valid_actions_mask.bool()
            # 1. Forza i logit delle azioni non valide a un valore molto basso
            logits = logits.masked_fill(~mask, -1e9)
            probs = F.softmax(logits, dim=-1)
            
            # 2. Uniform solo sui tasti validi per non sprecare esplorazione
            num_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = mask.float() / num_valid
        else:
            # Caso di fallback se non c'è maschera
            probs = F.softmax(logits, dim=-1)
            uniform = torch.full_like(probs, 1.0 / self.num_dist)

        # 3. Applica Unimix UNA SOLA VOLTA
        probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        
        dist = OneHotCategorical(probs=probs)
        action = dist.sample()
        
        # Calcolo entropia (sarà limitata a ln(num_valid))
        p_log_p = probs * torch.log(probs + 1e-8)
        entropy = -p_log_p.sum(dim=-1)

        return action, dist, entropy

    



    

