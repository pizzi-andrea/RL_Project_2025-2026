import cv2 as cv
from torch.nn.functional import sigmoid
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Any, List, Optional, Tuple, Union
from collections import deque

def critic_loss(v_hat_log, target_twohot, c_soft):
    """
    🎯 FIX DERIVATO DA RAY: La loss si occupa solo della classificazione dei bucket.
    La regolarizzazione avviene direttamente nello spazio dei parametri del network.
    """
    v_hat_log_probs = F.log_softmax(v_hat_log, dim=-1)
    
    per_step_loss = -(target_twohot * v_hat_log_probs).sum(dim=-1)
    weighted_loss = per_step_loss * c_soft
    
    lengths = c_soft.sum(dim=1).clamp(min=1.0)
    final_loss = weighted_loss.sum(dim=1) / lengths
    
    return final_loss.mean()


def actor_loss(td_return, critic_values, action_log_prob, entropy, ema_scaler, c_soft, entropy_scale=3e-4, mask=None):
    offset, invscale = ema_scaler(td_return, mask)
    
    scaled_targets = (td_return - offset) / invscale
    scaled_preds = (critic_values - offset) / invscale
  
    advantage = (scaled_targets - scaled_preds).detach()
    
    per_step_objective = (advantage * action_log_prob) + (entropy_scale * entropy)

    weighted_objective = per_step_objective * c_soft
    
    lengths = c_soft.sum(dim=1).clamp(min=1.0)
    final_loss = -( (weighted_objective.sum(dim=1) / lengths).mean() )
    
    with torch.no_grad():
        adv_metric = ((advantage * action_log_prob * c_soft).sum(dim=1) / lengths).mean().item()
        ent_metric = ((entropy_scale * entropy * c_soft).sum(dim=1) / lengths).mean().item()
        ema_val = invscale.mean().item()

    return final_loss, ema_val, adv_metric, ent_metric