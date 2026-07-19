import torch
import torch.nn.functional as F

import torch.distributions as D
def select_data(tensor, is_first, burn_in, T_train):
                # Se is_first: prendi [0 : T_train]
                # Se non is_first: prendi [burn_in : burn_in + T_train]
                out = []
                for b in range(tensor.shape[0]):
                    start = 0 if is_first[b] else burn_in
                    out.append(tensor[b, start : start + T_train])
                return torch.stack(out)

def kl_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, unimix: float = 0.01) -> torch.Tensor:
    """
    KL(q || p) for categorical distribution with 1% Uniform Mixing.
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


def symlog(x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def _unitwise_norm(x):
    if x.ndim <= 1:
        return x.norm(2)
    # Per i pesi convoluzionali o lineari, calcoliamo la norma per ogni unità di output
    return x.norm(2, dim=tuple(range(1, x.ndim)), keepdim=True)

@torch.no_grad()
def AGC(parameters, clip_factor=0.01, eps=1e-3):
    for p in parameters:
        if p.grad is None or p.ndim < 2: # Escludiamo bias e parametri 1D
            continue
            
        p_data = p.detach()
        g_data = p.grad.detach()
        
        p_norm = _unitwise_norm(p_data).clamp_(min=eps)
        g_norm = _unitwise_norm(g_data)
        
        max_norm = p_norm * clip_factor
        
        # Applichiamo il clipping solo dove la norma del gradiente supera la soglia
        trigger = g_norm > max_norm
        scaled_g = g_data * (max_norm / g_norm.clamp(min=eps))
        
        p.grad.copy_(torch.where(trigger, scaled_g, g_data))



def symexp(x):
    #x = x.clamp(mm, mx)
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1)

@torch.no_grad()
def twohot(x, B):
    """
    Codifica Two-Hot robusta per bucket equidistanti.
    
    Args:
        x: Target scalari (symlog), shape [Batch, T]
        B: Centri dei bucket, shape [Num_Buckets] (assunto linspace ordinato)
    Returns:
        out: Distribuzione Two-Hot, shape [Batch, T, Num_Buckets]
    """
    # 1. Calcoliamo lo step dei bucket (assumendo linspace)
    
    #x = symlog(x)
    min_b = B[0]
    max_b = B[-1]
    num_buckets = B.shape[0]
    step = (max_b - min_b) / (num_buckets - 1)

    # 2. Normalizziamo x nello spazio degli indici [0, Num_Buckets-1]
    # Formula inversa di linspace: index = (value - min) / step
    idx = (x - min_b) / step
    
    # 3. Clampiamo gli indici per evitare errori fuori range
    # Togliamo un epsilon (1e-5) per essere sicuri che l'indice massimo non sfori
    idx = idx.clamp(0, num_buckets - 1 - 1e-5)
    
    # 4. Troviamo gli indici interi adiacenti (Lower e Upper)
    lower = idx.floor().long().clamp(0, num_buckets - 2)
    upper = (lower + 1).clamp(0, num_buckets - 1)
    
    upper_weight = (idx - lower.float()).clamp(0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    
  
    
    # 6. Creiamo il tensore di output vuoto
    # Shape: [Batch, T, Num_Buckets]
    out = torch.zeros(*x.shape, num_buckets, device=x.device)
    
    # 7. Distribuiamo i pesi (Scatter)
    # Dobbiamo espandere l'ultima dimensione per fare matching con 'out' durante lo scatter
    # lower.unsqueeze(-1) trasforma [B, T] in [B, T, 1]
    out.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    out.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    
    return out