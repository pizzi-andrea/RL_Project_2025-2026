import torch
import torch.nn.functional as F

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
    q_logprob = torch.log(q_prob)
    p_logprob = torch.log(p_prob)
    
    # Standard KL calculation: sum(q * (log_q - log_p))
    kl = (q_prob * (q_logprob - p_logprob)).sum(dim=-1)
    
    return kl

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


def symlog(x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)



def loss_prediction(x_hat, x, r_hat, r, c_hat, c, att_hat, att, mask=None, 
                    beta=1, beta_x=1.0, beta_r=1.0, beta_c=1.0, beta_att=1.0):
    """
    Ritorna la loss di predizione per ogni passo temporale [B, T]
    """
    # 1. Obs Loss [B, T] (Assumendo x normalizzato [0,1] per BCE, altrimenti usa MSE)
    # reduction='none' è fondamentale per mantenere [B, T]
    loss_x = F.binary_cross_entropy_with_logits(x_hat, x, reduction='none').mean(dim=(2, 3, 4)) 
    
    # 2. Reward Loss [B, T]
    loss_r = F.mse_loss(r_hat, symlog(r), reduction='none')
    
    # 3. Continue Loss [B, T]
    loss_c = F.binary_cross_entropy_with_logits(c_hat, c, reduction='none')
    
    # 4. Attribute Loss [B, T]
    loss_att = F.mse_loss(att_hat, symlog(att), reduction='none').mean(dim=-1)

    # Somma pesata
    loss_total_bt = (loss_x * beta_x) + (loss_r * beta_r) + (loss_c * beta_c) + (loss_att * beta_att)

    # Applica mask (mette a 0 i passi non validi)
    if mask is not None:
        loss_total_bt = loss_total_bt * mask

    return loss_total_bt * beta # Shape [B, T]

def loss_dynamic(q_log, p_log, mask=None, beta=1.0):
    """
    Ritorna la loss KL dinamica per ogni passo temporale [B, T]
    """
    q_log = q_log.detach() # Stop gradient sul posterior

    # 1. Calcola KL [B, T]
    kl_bt = kl_logits(q_log, p_log).sum(dim=-1)
    
    
    loss_bt = kl_bt
    
    # 3. Masking
    if mask is not None:
        loss_bt = loss_bt * mask
        
    return loss_bt*beta # Shape [B, T]

def loss_representation(q_log, p_log, mask=None, beta=0.1):
    """
    Ritorna la loss KL representation per ogni passo temporale [B, T]
    """
    p_log = p_log.detach() # Stop gradient sul prior

    kl_bt = kl_logits(q_log, p_log).sum(dim=-1)
    
    loss_bt = kl_bt 
    
    if mask is not None:
        loss_bt = loss_bt * mask
        
    return loss_bt*beta # Shape [B, T]