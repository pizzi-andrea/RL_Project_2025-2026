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
    Critic loss based on twohot encoding for high variance targets,
    both values v_hat_logs and target_twohot muste be in symlog space
    """

    # == transform raw logits in probs
    v_hat_log_probs = F.log_softmax(v_hat_log, dim=-1)
    
    # == compute cross entropy in symlog space (over buckets)
    per_step_loss = -(target_twohot * v_hat_log_probs).sum(dim=-1)
    weighted_loss = per_step_loss * c_soft
    
    # == aggregate results ignoring padding
    lengths = c_soft.sum(dim=1).clamp(min=1.0)
    final_loss = weighted_loss.sum(dim=1) / lengths
    
    return final_loss.mean()


def actor_loss(td_return, critic_values, action_log_prob, entropy, ema_scaler, mask, entropy_scale=3e-4, ):
    """"Normalize actor loss suitable for multi-enviroment rewards"""

    # == normalize td_return to avoid gradient explosion 
    offset, invscale = ema_scaler(td_return, mask)
    
    # == scale all values 
    scaled_targets = (td_return - offset) / invscale
    scaled_preds = (critic_values - offset) / invscale
    
    # == compute normalized avantages 
    advantage = (scaled_targets - scaled_preds).detach()
    
    # == compute loss value
    per_step_objective = (advantage * action_log_prob) + (entropy_scale * entropy)

    weighted_objective = per_step_objective * mask
    
    lengths = mask.sum(dim=1).clamp(min=1.0)
    final_loss = -( (weighted_objective.sum(dim=1) / lengths).mean() )
    
    # == update EMA scaler
    with torch.no_grad():
        adv_metric = ((advantage * action_log_prob * mask).sum(dim=1) / lengths).mean().item()
        ent_metric = ((entropy_scale * entropy * mask).sum(dim=1) / lengths).mean().item()
        ema_val = invscale.mean().item()

    return final_loss, ema_val, adv_metric, ent_metric