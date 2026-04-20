import torch
import torch.nn.functional as F

def select_data(tensor, is_first, burn_in, T_train):
                # Se is_first: prendi [0 : T_train]
                # Se non is_first: prendi [burn_in : burn_in + T_train]
                out = []
                for b in range(tensor.shape[0]):
                    start = 0 if is_first[b] else burn_in
                    out.append(tensor[b, start : start + T_train])
                return torch.stack(out)

def kl_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, unimix: float = 0.01, free_nats:float=1.0) -> torch.Tensor:
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


    kl = (q_prob * (q_logprob - p_logprob)) # [B, T, C]

    kl = kl.sum(dim=-1) 
    kl = kl.sum(dim=-1) 

   
    if free_nats > 0:
        kl = torch.maximum(kl, torch.tensor(free_nats, device=kl.device))
    
    return  kl

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

def _unitwise_norm(x):
    if x.ndim <= 1:
        return x.norm(2)
    # Per i pesi convoluzionali o lineari, calcoliamo la norma per ogni unità di output
    return x.norm(2, dim=tuple(range(1, x.ndim)), keepdim=True)

def AGC(parameters, clip_factor=0.01, eps=1e-3):
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        
        p_norm = _unitwise_norm(p_data).clamp_(min=eps)
        g_norm = _unitwise_norm(g_data)
        
       
        max_norm = p_norm * clip_factor
        
        
        trigger = g_norm > max_norm
        
        scaled_g = g_data * (max_norm / g_norm.clamp(min=1e-6))
        p.grad.data.copy_(torch.where(trigger, scaled_g, g_data))

def loss_prediction(x_hat, x, r_hat, r, c_hat, c, att_hat, att, mask=None, 
                    beta_x=1, beta_r=1.0, beta_c=1, beta_att=1):
    """
    Prediction loss (eq x. paper)
    """
    # loss for observation (graphical)
    #x_hat = F.sigmoid(x_hat)
    loss_x = F.binary_cross_entropy_with_logits(x_hat, x, reduction="none").mean(dim=(2,3,4))
    loss_c = F.binary_cross_entropy_with_logits(c_hat, c, reduction='none')
    # 2. Reward Loss  

    loss_r = reward_loss(r_hat, r, mask)
    loss_att = symlog_loss(att_hat, att, mask)

    if mask is not None:
         loss_c = loss_c * mask
         loss_x = loss_x * mask
    
    
    
    loss_total_bt = (loss_x * beta_x) + (loss_r * beta_r) + (loss_c * beta_c) + (loss_att * beta_att)
    
    loss_att_rec = (loss_att * beta_att).detach().cpu()
    loss_c_rec = (loss_c * beta_c).detach().cpu()
    loss_x_rec = (loss_x * beta_x).detach().cpu()
    loss_reward_rec = (loss_r * beta_r).detach().cpu()
    
    
    
    # weighted sum by sequence length
    lengths = mask.sum(dim=1).clamp(min=1).detach().cpu()


    components = {
        "obs_loss"      : (loss_x_rec.sum(-1)/ lengths).mean().item(),
        "reward_loss"   : (loss_reward_rec.sum(-1)/ lengths).mean().item(),
        "continue_loss" : (loss_c_rec.sum(-1)/ lengths).mean().item(),
        "attribute_loss": (loss_att_rec.sum(-1)/ lengths).mean().item(),
    }
    return loss_total_bt, components # Shape [B, T]

def loss_dynamic(q_log, p_log, mask=None, unimix:float=0.01, free_nats:float=1.0):
    """
    Compute loss value for dynamic model (prior)
    """
    q_log = q_log.detach() # Stop gradient sul posterior

  
    kl_bt = kl_logits(q_log, p_log, unimix, free_nats) # [B,T]
    
    if mask is not None:
        kl_bt = kl_bt * mask

    
        
    return kl_bt

def reward_loss(r_hat, r, mask=None):
    log_probs = torch.nn.functional.log_softmax(r_hat, dim=-1)
    per_step_loss = -(r * log_probs).sum(-1)

    if mask is not None:
        per_step_loss = per_step_loss * mask    
    
    return per_step_loss

def loss_representation(q_log, p_log, mask=None, unimix:float=0.01, free_nats:float=1.0):
    """
    (posterior)
    """
    p_log = p_log.detach()

    kl_bt = kl_logits(q_log, p_log, unimix, free_nats)
    
    if mask is not None:
        kl_bt = kl_bt * mask

    return kl_bt

def symlog_loss(pred_symlog, real_symlog, mask=None):
    """Evaluate the network capacity to predict target value directly in symlog scale"""
    loss = 0.5*torch.pow((pred_symlog - real_symlog), 2).sum(-1)
    if mask is not None:
        loss = loss*mask
    
    return loss


def symexp(x, mm=-20.0, mx=20.0):
    #x = x.clamp(mm, mx)
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1)

@torch.no_grad()
def twohot(x, B):
    """
    Robust twohot encoding
    
    Args:
        x: uncoded input
        B: Centri dei bucket, shape [Num_Buckets] (assunto linspace ordinato)
    Returns:
        out: Distribuzione Two-Hot, shape [Batch, T, Num_Buckets]
    """
    # 1. Calcoliamo lo step dei bucket (assumendo linspace)
    x = symlog(x)
    min_b = B[0]
    max_b = B[-1]
    num_buckets = B.shape[0]
    step = (max_b - min_b) / (num_buckets - 1)

    # 2. Normalizziamo x nello spazio degli indici [0, Num_Buckets-1]
    # Formula inversa di linspace: index = (value - min) / step
    idx = (x - min_b) / step
    
    # 3. Clampiamo gli indici per evitare errori fuori range
    # Togliamo un epsilon (1e-5) per essere sicuri che l'indice massimo non sfori
    #idx = idx.clamp(0, num_buckets - 1 - 1e-5)
    
    # 4. Troviamo gli indici interi adiacenti (Lower e Upper)
    lower = idx.floor().long()
    upper = lower + 1
    
    # 5. Calcoliamo i pesi (interpolazione lineare)
    # Se idx è 5.7 -> lower=5, upper=6.
    # Peso su upper (6) = 0.7
    # Peso su lower (5) = 1.0 - 0.7 = 0.3
    upper_weight = idx - lower.float()
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