import torch
import torch.nn.functional as F
import torch.distributions as D

def select_data(tensor, is_first, burn_in, T_train):
    """Utility metods for traning procedure"""
    out = []
    for b in range(tensor.shape[0]):
        start = 0 if is_first[b] else burn_in
        out.append(tensor[b, start : start + T_train])
    return torch.stack(out)

def kl_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, unimix: float = 0.01) -> torch.Tensor:
    """
    KL(q || p) for categorical distribution with 1% Uniform Mixing.

    Args:
        q_logits: logits about posterior distribution
        p_logits: logits about prior distribution
        unimix: uniform addistribution mix to prevent probs collapse
    """
    # Get probabilities from logits
    q_prob = F.softmax(q_logits, dim=-1)
    p_prob = F.softmax(p_logits, dim=-1)

    # Apply Uniform Mixing (1% uniform + 99% network output) to prevent NaN and collapse
    n_classes = q_logits.shape[-1]
    if unimix > 0.0:
        q_prob = (1.0 - unimix) * q_prob + (unimix / n_classes)
        p_prob = (1.0 - unimix) * p_prob + (unimix / n_classes)

    # Compute log-probabilities from mixed probabilities
    q = D.Independent(
            D.OneHotCategorical(probs=q_prob), 1, True
    )
    p  = D.Independent(
            D.OneHotCategorical(probs=p_prob), 1, True
    )

    return  D.kl_divergence(q, p)




def _unitwise_norm(x):
    if x.ndim <= 1:
        return x.norm(2)
    return x.norm(2, dim=tuple(range(1, x.ndim)), keepdim=True)

@torch.no_grad()
def AGC(parameters, clip_factor=0.01, eps=1e-3):
    """Adaptive Gradient Clipping"""
    for p in parameters:
        if p.grad is None or p.ndim < 2: 
            continue
            
        p_data = p.detach()
        g_data = p.grad.detach()
        
        p_norm = _unitwise_norm(p_data).clamp_(min=eps)
        g_norm = _unitwise_norm(g_data)
        
        max_norm = p_norm * clip_factor
        
        
        trigger = g_norm > max_norm
        scaled_g = g_data * (max_norm / g_norm.clamp(min=eps))
        
        p.grad.copy_(torch.where(trigger, scaled_g, g_data))


def symlog(x):
        """symlog operation presented in DreaamerV3 paper"""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    """Inverse of symlog operator"""
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1.0)

@torch.no_grad()
def twohot(x, B):
    """
    Encode x value in symmetric and equidistanciated buckets
    
    Args:
        x: Scalar target (symlog), shape [Batch, T]
        B: Number of buckets
    Returns:
        out: Distribuzione Two-Hot, shape [Batch, T, Num_Buckets]
    """
    # 1. Calcoliamo lo step dei bucket (assumendo linspace)
    
    #x = symlog(x)
    min_b = B[0]
    max_b = B[-1]
    num_buckets = B.shape[0]
    step = (max_b - min_b) / (num_buckets - 1)
    idx = (x - min_b) / step
    

    idx = idx.clamp(0, num_buckets - 1 - 1e-5)
    lower = idx.floor().long().clamp(0, num_buckets - 2)
    upper = (lower + 1).clamp(0, num_buckets - 1)
    
    upper_weight = (idx - lower.float()).clamp(0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    
    out = torch.zeros(*x.shape, num_buckets, device=x.device)
    out.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    out.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    
    return out