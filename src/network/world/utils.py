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

def loss_prediction(x_hatlog, r_hatlog, c_hatlog,  att_hatlog, x, r, c, att, beta=1, beta_x=1, beta_r=1, beta_c=1, beta_att=1, mask=None):
    """Loss for recostruction"""

    # loss for obs
    loss_x = F.binary_cross_entropy_with_logits(x_hatlog, x, reduction="none")
    loss_xt = loss_x.mean(dim=(2, 3, 4))
    
    # loss for rewards
    loss_r = F.mse_loss(r_hatlog, symlog(r), reduction="none")
    loss_rt = loss_r

    # loss for continus flags
    loss_c = F.binary_cross_entropy_with_logits(c_hatlog, c, reduction="none")
    loss_ct = loss_c

    # loss for attributes
    loss_att = F.mse_loss(att_hatlog, symlog(att), reduction="none")
    loss_attt = loss_att.mean(dim=-1)


    loss_pred = +(loss_xt*beta_x + loss_rt*beta_r + loss_ct*beta_c + loss_attt*beta_att)

    if mask is not None:
         loss_t = loss_pred * mask
         loss = loss_t.sum() / torch.clamp(mask.sum(), min=1.0)
    else:
         loss_t = loss_pred
         loss = loss_t.mean(dim=-1).mean()

    loss = beta*loss
    return loss, loss_t

def loss_dynamic(q_log, p_log, mask=None, beta=1):
    """Loss for dynamic world"""
    q_log = q_log.detach()

    # compute KL divergence
    loss_dynamic_t_raw = kl_logits(q_log, p_log).sum(dim=-1)
    
    # hard clamp
    nats = torch.tensor(1.0, device=q_log.device, dtype=q_log.dtype)
    loss_dynamic_t = torch.max(loss_dynamic_t_raw, nats)
    #loss_dynamic_t = loss_dynamic * beta 

    # masking
    if mask is not None:
        loss_dynamic_t = loss_dynamic_t * mask
        loss_dynamic_t_raw = loss_dynamic_t_raw*mask
        
        
        loss_dynamic = loss_dynamic_t.sum() / torch.clamp(mask.sum(), min=1.0)
        loss_dynamic_raw = (loss_dynamic_t_raw.sum() / torch.clamp(mask.sum(), min=1.0)).detach().cpu().item()
    else:
        
        loss_dynamic = loss_dynamic_t.mean()
        loss_dynamic_raw = loss_dynamic_t_raw.mean().detach().cpu().item()
    
    loss_dynamic = loss_dynamic*beta
    loss_dynamic_raw = loss_dynamic_raw*beta

    return loss_dynamic, loss_dynamic_t, loss_dynamic_raw

def loss_representation(q_log, p_log, mask=None, beta=0.1):
    """Loss for reppresentation world"""
    p_log = p_log.detach()

    # compute KL divergence 
    loss_representation_t_raw = kl_logits(q_log, p_log).sum(dim=-1)
    
    # hard clamp
    nats = torch.tensor(1.0, device=q_log.device, dtype=q_log.dtype)
    loss_representation_t = torch.max(loss_representation_t_raw, nats)
    #loss_representation_t = loss_representation * beta 

    # masking
    if mask is not None:
        loss_representation_t = loss_representation_t * mask
        loss_representation_t_raw = loss_representation_t_raw*mask
        
        
        loss_representation = loss_representation_t.sum() / torch.clamp(mask.sum(), min=1.0)
        loss_representation_raw = (loss_representation_t_raw.sum() / torch.clamp(mask.sum(), min=1.0)).detach().cpu().item()
    else:
        
        loss_representation = loss_representation_t.mean()
        loss_representation_raw = loss_representation_t_raw.mean().detach().cpu().item()
    
    loss_representation = loss_representation*beta

    return loss_representation, loss_representation_t, loss_representation_raw