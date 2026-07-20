import torch
import torch.nn.functional as F
from .utils import kl_logits

def image_loss(x_logits, x_target):
    x_logits = torch.sigmoid(x_logits)
    return torch.sum(torch.square(x_logits - x_target), dim=[-3, -2, -1]) * 0.5


def loss_prediction(x_hat, x, r_hat, r, c_hat, c, att_hat, att, mask=None, 
                    beta_x=1, beta_r=1.0, beta_c=1, beta_att=1):
    """
    Prediction loss (eq x. paper)
    """
    
    # loss for images
    loss_x = image_loss(x_hat, x)
    # loss continue
    loss_c = F.binary_cross_entropy_with_logits(c_hat, c, reduction='none')

    # loss rewards
    loss_r = reward_loss(r_hat, r)

    # loss attributes
    loss_att = symlog_loss(att_hat, att)

    if mask is not None:
         loss_x = loss_x * mask
         loss_c = loss_c * mask
         loss_r = loss_r * mask
         loss_att = loss_att * mask
    
    
    
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

def loss_dynamic(q_log, p_log, mask=None, unimix:float=0.01):
    """
    Compute loss value for dynamic model (prior)
    """
    q_log = q_log.detach() # Stop gradient sul posterior

  
    kl_bt = kl_logits(q_log, p_log, unimix) # [B,T]
    
    if mask is not None:
        kl_bt = kl_bt * mask
        
    return kl_bt

def reward_loss(r_hat, r):
    log_probs = torch.nn.functional.log_softmax(r_hat, dim=-1)
    per_step_loss = -(r * log_probs).sum(-1)

    return per_step_loss

def loss_representation(q_log, p_log, mask=None, unimix:float=0.01):
    """
    """
    p_log = p_log.detach()

    kl_bt = kl_logits(q_log, p_log, unimix)
    
    if mask is not None:
        kl_bt = kl_bt * mask

    return kl_bt

def symlog_loss(pred_symlog, real_symlog):
    """Evaluate the network capacity to predict target value directly in symlog scale"""
    loss = 0.5*torch.pow((pred_symlog - real_symlog), 2).sum(-1)
    
    return loss