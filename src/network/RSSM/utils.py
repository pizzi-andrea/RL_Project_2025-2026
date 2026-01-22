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

