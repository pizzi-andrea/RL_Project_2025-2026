import torch
import torch.nn.functional as F

def cat_kl_logits(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    # KL(q || p) for categorical, computed from logits
    q_logprob = F.log_softmax(q_logits, dim=-1) #posterior
    p_logprob = F.log_softmax(p_logits, dim=-1) #prior
    q_prob = torch.softmax(q_logits, dim=-1)
    return (q_prob * (q_logprob - p_logprob)).sum(dim=-1) #discrete KL

def last_valid_from_mask(x_bt: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    x_bt: [B,T]
    mask_bt: [B,T] (1 valid, 0 padding)
    return: [B] x at last valid timestep
    """
    B, T = x_bt.shape
    lengths = mask_bt.sum(dim=1).long()                 # [B]
    last_idx = (lengths - 1).clamp_min(0)               # [B]
    ar = torch.arange(B, device=x_bt.device)
    return x_bt[ar, last_idx]
