import torch
import torch.nn.functional as F



def unimix_probs(probs: torch.Tensor, unimix: float = 0.01) -> torch.Tensor: #uniform noise addition
    # probs: [..., K]
    K = probs.shape[-1]
    return (1 - unimix) * probs + unimix * (torch.ones_like(probs) / K)

def st_onehot(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
def logits2discrate(logits: torch.Tensor, L:int, K:int, unimix=0.01):
    logits_pure = logits.view(-1, L, K) 
    probs = torch.softmax(logits_pure, dim=-1)
    
    if unimix > 0.0:
        probs = unimix_probs(probs, unimix=unimix)
    
    logits_mix = torch.log(probs.clamp_min(1e-8))
    z_st = st_onehot(logits_mix)
    z_flat = z_st.view(-1, L * K)
    
    return z_flat, logits_pure # 