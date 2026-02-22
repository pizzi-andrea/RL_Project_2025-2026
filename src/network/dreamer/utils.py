import torch
import torch.nn.functional as F



def unimix_probs(probs: torch.Tensor, unimix: float = 0.01) -> torch.Tensor: #uniform noise addition
    # probs: [..., K]
    K = probs.shape[-1]
    return (1 - unimix) * probs + unimix * (torch.ones_like(probs) / K)

def st_onehot(logits_mix: torch.Tensor) -> torch.Tensor:
   
    probs = torch.softmax(logits_mix, dim=-1)
    index = probs.argmax(dim=-1, keepdim=True)
    hard_onehot = torch.zeros_like(probs).scatter_(-1, index, 1.0)
   
    return hard_onehot + probs - probs.detach() # trick
def logits2discrate(logits: torch.Tensor, unimix=0.01):
    L, K = logits.shape[-2], logits.shape[-1]
    
    probs = torch.softmax(logits, dim=-1)
    
    if unimix > 0.0:
        probs = unimix_probs(probs, unimix=unimix)
    
    logits_mix = torch.log(probs)
    z_st = st_onehot(logits_mix)
    z_flat = z_st.view(-1, L * K)
    
    return z_flat # 