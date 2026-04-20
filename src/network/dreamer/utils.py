import torch
import torch.nn.functional as F



def unimix_probs(probs: torch.Tensor, unimix: float = 0.01) -> torch.Tensor: #uniform noise addition
    # probs: [..., K]
    K = probs.shape[-1]
    return (1 - unimix) * probs + (unimix / K)

def st_onehot(probs: torch.Tensor) -> torch.Tensor:
    """One-hot categorical distribution"""
   
    
    index = probs.argmax(dim=-1, keepdim=True)
    hard_onehot = torch.zeros_like(probs).scatter_(-1, index, 1.0)
    
    # make differentiable 
    return hard_onehot + probs - probs.detach() 


def logits2discrate(logits: torch.Tensor, unimix=0.01):
    
    
    # convert logits in categoric rappresentation
    probs = torch.softmax(logits, dim=-1)
    
    # add unimax and prevent prob collapse
    if unimix > 0.0:
        probs = unimix_probs(probs, unimix=unimix)
    
    z_st = st_onehot(probs)
    
    return z_st # 