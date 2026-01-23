import torch
import torch.nn.functional as F



def unimix_probs(probs: torch.Tensor, unimix: float = 0.01) -> torch.Tensor: #uniform noise addition
    # probs: [..., K]
    K = probs.shape[-1]
    return (1 - unimix) * probs + unimix * (torch.ones_like(probs) / K)

def st_onehot_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Straight-through argmax one-hot sample from categorical logits
    # logits: [..., K]
    with torch.no_grad():
        idx = torch.argmax(logits, dim=-1)
        y_hard = F.one_hot(idx, num_classes=logits.shape[-1]).float() #1hot vector that represents the real choice of the latent state
    probs = torch.softmax(logits, dim=-1)
    # straight-through: forward discrete, backward continuous
    return y_hard + (probs - probs.detach()) # this is equal to y_hard in forward, but gradients are from probs in the backward pass

def cat_kl_logits(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    # KL(q || p) for categorical, computed from logits
    q_logprob = F.log_softmax(q_logits, dim=-1) #posterior
    p_logprob = F.log_softmax(p_logits, dim=-1) #prior
    q_prob = torch.softmax(q_logits, dim=-1)
    return (q_prob * (q_logprob - p_logprob)).sum(dim=-1) #discrete KL