import torch
import torch.nn.functional as F


def cat_kl_logits(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    # KL(q || p) for categorical, computed from logits
    q_logprob = F.log_softmax(q_logits, dim=-1) #posterior
    p_logprob = F.log_softmax(p_logits, dim=-1) #prior
    q_prob = torch.softmax(q_logits, dim=-1)
    return (q_prob * (q_logprob - p_logprob)).sum(dim=-1) #discrete KL