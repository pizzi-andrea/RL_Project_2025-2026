import torch
import torch.nn.functional as F



def unimix_probs(probs: torch.Tensor, unimix: float = 0.01) -> torch.Tensor: #uniform noise addition
    # probs: [..., K]
    K = probs.shape[-1]
    return (1 - unimix) * probs + unimix * (torch.ones_like(probs) / K)

def st_onehot(logits: torch.Tensor) -> torch.Tensor:
    
    #  Straight-through
    
    probs = torch.softmax(logits, dim=-1)
    
    with torch.no_grad():
        
        shape = probs.shape
        probs_flat = probs.view(-1, shape[-1])
        idx = torch.multinomial(probs_flat, num_samples=1).view(shape[:-1])
        
        # one-hot encoding
        y_hard = F.one_hot(idx, num_classes=shape[-1]).float()
    
    return y_hard + (probs - probs.detach())
def logits2discrate(self, logits: torch.Tensor, L:int, K:int) -> tuple[torch.Tensor, torch.Tensor]: #conversion logit-> discrete z
        # logits: [B,L*K] -> [B,L,K]
        logits = logits.view(-1, L, K)
        probs = torch.softmax(logits, dim=-1) #softmax to obtain a probability
        probs = unimix_probs(probs, unimix=self.unimix) #unimix that adds some uniform noise
        # incorporate unimix back into logits (stable)
        logits = torch.log(probs.clamp_min(1e-8)) # stable logits after unimix
        z_st = st_onehot(logits)  # [B,L,K] z discrete but derivable
        z_flat = z_st.view(-1, self.L * self.K)
        return z_flat, logits