import cv2 as cv
from torch.nn.functional import sigmoid
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Any, List, Optional, Tuple, Union
from collections import deque
class EMAScalar(nn.Module):
    def __init__(self, decay=0.98, eps=1e-6):
        super().__init__()
        self.decay = decay
        self.eps = eps

        self.register_buffer("pct5", torch.tensor(float('nan')))
        self.register_buffer("pct95", torch.tensor(float('nan')))

    def forward(self, values, mask=None):
        values = values.detach()
        
        if not self.training:
            if torch.isnan(self.pct5):
                return 0.0, 1.0
            return self.pct5.detach(), torch.clamp(self.pct95 - self.pct5, min=self.eps).detach()

        with torch.no_grad():
            flat = values[mask.bool()] if mask is not None else values.flatten()
            cur_5 = torch.quantile(flat, 0.05)
            cur_95 = torch.quantile(flat, 0.95)

            if torch.isnan(self.pct5):
                self.pct5.copy_(cur_5)
                self.pct95.copy_(cur_95)
            else:
                self.pct5.mul_(self.decay).add_((1.0 - self.decay) * cur_5)
                self.pct95.mul_(self.decay).add_((1.0 - self.decay) * cur_95)

        invscale = torch.clamp(self.pct95 - self.pct5, min=self.eps)
        return self.pct5.detach(), invscale.detach()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    #x = x.clamp(mm, mx)
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1.0)


# utils.py modificato
# Sostituisci la tua funzione lambda_returns in utils.py
@torch.no_grad()
def lambda_returns(rewards, expected_values, execution_mask, gamma=0.997, lam=0.95):
    """
    rewards: [B, T]
    expected_values: [B, T+1] (contiene il frame finale di bootstrap)
    execution_mask: [B, T] (Probabilità 'continue' o maschera reale)
    """
    returns = torch.zeros_like(expected_values)
    
    # 1. Bootstrap finale (preso dal frame T+1)
    last_ret = expected_values[:, -1]
    returns[:, -1] = last_ret 
    
    # 2. Propagazione a ritroso
    for t in reversed(range(expected_values.shape[1] - 1)):
        # Il target futuro si calcola mixando il valore atteso e il ritorno precedente
        target = (1 - lam) * expected_values[:, t + 1] + lam * last_ret
        
        # LA SOLUZIONE: execution_mask moltiplica SOLO il target futuro.
        # La reward corrente rimane intatta, sia essa +1, -10 o 0.
        returns[:, t] = rewards[:, t] + gamma * execution_mask[:, t] * target
        
        last_ret = returns[:, t]
        
    return returns[:, :-1]


# Sostituisci la tua funzione critic_loss in utils.py
def critic_loss(v_hat_log, target_twohot, c_soft, slow_critic_twohot=None):
    """
    c_soft: valid_mask (o discount) [B, T]
    """
    per_step_loss = -(target_twohot * v_hat_log).sum(dim=-1)

    if slow_critic_twohot is not None:
        reg_loss = -(slow_critic_twohot * v_hat_log).sum(dim=-1)
        per_step_loss = per_step_loss + reg_loss
    
    weighted_loss = per_step_loss * c_soft
    
    lengths = c_soft.sum(dim=1).clamp(min=1.0)
    final_loss = weighted_loss.sum(dim=1) / lengths
    
    return final_loss.mean()


def make_video(frames, output_path, fps=30):
    frames = frames.cpu().detach()
    frames_video = (frames*255).to(dtype=torch.uint8)
    frames_video = frames_video.permute(0,2,3,1).numpy()
    frames_video = frames_video[:, :, :, ::-1]
    num_frames, height, width, _ = frames_video.shape
    fourcc = cv.VideoWriter_fourcc(*'XVID') # Codec per file .mp4
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 6. Scrittura dei frame
    for i in range(num_frames):
        video_writer.write(frames_video[i])
    
    # 7. Rilascio risorse
    video_writer.release()
    print(f"new video saved in : {output_path}")

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
        p.grad.copy_(torch.where(trigger, scaled_g, g_data))


def actor_loss(td_return, critic_values, action_log_prob, entropy, ema_scaler, c_soft, entropy_scale=3e-4, mask=None):
    """
    c_soft: valid_mask [B, T] (1 per dati reali, 0 per padding)
    """
    # FIX: Comprimiamo i valori nello spazio Symlog 
    offset, invscale = ema_scaler(td_return, mask)
    
    scaled_targets = (td_return - offset) / invscale
    scaled_preds = (critic_values - offset) / invscale
  
    advantage = (scaled_targets - scaled_preds).detach()
    
    per_step_objective = (advantage * action_log_prob) + (entropy_scale * entropy)

    # 4. Applicazione MASCHERA (c_soft)
    weighted_objective = per_step_objective * c_soft
    
    # 5. Media pesata corretta
    lengths = c_soft.sum(dim=1).clamp(min=1.0)
    final_loss = -( (weighted_objective.sum(dim=1) / lengths).mean() )
    
    # Metriche
    with torch.no_grad():
        adv_metric = ((advantage * action_log_prob * c_soft).sum(dim=1) / lengths).mean().item()
        ent_metric = ((entropy_scale * entropy * c_soft).sum(dim=1) / lengths).mean().item()
        ema_val = invscale.mean().item()

    return final_loss, ema_val, adv_metric, ent_metric

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
    x = x.clamp(min_b, max_b)
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


def force_list(
    elements: Optional[Any] = None, to_tuple: bool = False
) -> Union[List, Tuple]:
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single
    item, already a list, or a tuple.

    Args:
        elements: The inputs as a single item, a list/tuple/deque of items, or None,
            to be converted to a list/tuple. If None, returns empty list/tuple.
        to_tuple: Whether to use tuple (instead of list).

    Returns:
        The provided item in a list of size 1, or the provided items as a
        list. If `elements` is None, returns an empty list. If `to_tuple` is True,
        returns a tuple instead of a list.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return (
        ctor()
        if elements is None
        else ctor(elements)
        if type(elements) in [list, set, tuple, deque]
        else ctor([elements])
    )

def dreamerv3_normal_initializer(parameters):
    """From Danijar Hafner's DreamerV3 JAX repo.

    Used on any layer whenever the config for that layer has `winit="normal"`.

    Note: Not identical with Glorot normal. Differs in the std computation
    glorot_std = sqrt(2/(fanin+fanout))
    this_std = sqrt(1/AVG(fanin, fanout)) / [somemagicnumber=0.879...]
    """
    for param in force_list(parameters):
        if param.dim() > 1:
            fanin, fanout = _fans(param.shape)
            scale = 1.0 / np.mean([fanin, fanout])
            std = np.sqrt(scale) / 0.87962566103423978
            with torch.no_grad():
                param.normal_(0, std)
                param.clamp_(-2, 2)


def _fans(shape):
    if len(shape) == 0:
        return 1, 1
    elif len(shape) == 1:
        return shape[0], shape[0]
    elif len(shape) == 2:
        return shape
    else:
        space = int(np.prod(shape[:-2]))
        return shape[-2] * space, shape[-1] * space