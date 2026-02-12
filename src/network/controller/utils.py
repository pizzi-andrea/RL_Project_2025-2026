import cv2 as cv
import numpy as np
import torch 
from torch import nn
import torchvision.transforms.v2.functional as F
import torchvision.io as io
from PIL import Image
from torch import Tensor, no_grad
from torch.nn.functional import cross_entropy, sigmoid
class EMAScalar(nn.Module):
    def __init__(self, decay=0.98, eps=1e-6):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, values):
        if not self.training:
            return self.scale
        
        with torch.no_grad():
            per_95 = torch.quantile(values, 0.95)
            per_05 = torch.quantile(values, 0.05)
            range_val = (per_95 - per_05).clamp_min(self.eps)

            self.scale.mul_(self.decay).add_((1 - self.decay) * range_val)

        return self.scale

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

import torch

import torch

@torch.no_grad() 
def lambda_returns(rewards, values, continues, gamma=0.997, lam=0.95):
    """
    Calcola i Lambda-returns. Gestisce input di lunghezza uguale [B, T].
    Se len(values) == len(rewards), calcola T-1 ritorni (perde l'ultimo step).
    """
    # rewards:   [B, T]
    # values:    [B, T] oppure [B, T+1]
    
    # 1. Gestione del disallineamento T vs T+1
    if values.shape[1] == rewards.shape[1]:
        # Se abbiamo tanti valori quanti reward, non possiamo calcolare
        # il ritorno per l'ultimo reward perché manca il valore futuro (bootstrap).
        # Quindi ignoriamo l'ultimo reward e calcoliamo returns per 0..T-2.
        rewards = rewards[:, :-1]
        continues = continues[:, :-1]
        
        # values[:, :-1] sono i V(s_t) per t=0..T-2
        # values[:, 1:]  sono i V(s_{t+1}) per t=0..T-2 (che funge da bootstrap)
        v_pred = values[:, :-1]
        next_val = values[:, 1:]
    else:
        # Caso standard: values ha un elemento in più (bootstrap esplicito)
        # rewards: [B, T], values: [B, T+1]
        v_pred = values[:, :-1]
        next_val = values[:, 1:]

    batch_size, horizon = rewards.shape
    
    # Inizializza buffer
    last_gae = torch.zeros(batch_size, device=rewards.device)
    returns_list = []
    
    # Backward Pass
    for t in reversed(range(horizon)):
        r_t = rewards[:, t]
        c_t = continues[:, t]
        v_next = next_val[:, t]   # V(s_{t+1})
        v_curr = v_pred[:, t]     # V(s_t)
        
        # Delta TD
        delta = r_t + gamma * c_t * v_next - v_curr
        
        # GAE
        last_gae = delta + gamma * lam * c_t * last_gae
        
        # Lambda Return = Advantage + Value
        result_t = last_gae + v_curr
        
        returns_list.insert(0, result_t)
        
    targets = torch.stack(returns_list, dim=1)
    return targets





def make_video(frames, output_path, fps=30):
    frames = frames.cpu().detach()
    frames_video = (sigmoid(frames)*255).to(dtype=torch.uint8)
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




def actor_loss(td_return, critic_values, action_log_prob, entropy, ema_scaler, entropy_scale=3e-4, debug=False):
    advantages = td_return - critic_values
    s = ema_scaler(td_return)
    scaled_advantages = (advantages / torch.clamp(s, min=1.0)).detach()

    loss = -(scaled_advantages*action_log_prob + entropy_scale*entropy).mean()

    # --- DEBUG BLOCK START ---
    if debug:
        with torch.no_grad():
        
                print(f"\n--- DEBUG ACTOR LOSS ---")
                print(f"Returns (Target): Mean={td_return.mean().item():.4f}, Max={td_return.max().item():.4f}")
                print(f"Critic Values:    Mean={critic_values.mean().item():.4f}")
                print(f"Advantage Raw:    Mean={advantages.mean().item():.4f}")
                print(f"EMA Scaler (s):   Value={s.mean().item():.4f}")
                print(f"Log Probs:        Mean={action_log_prob.mean().item():.4f}")
                print(f"Entropy:          Mean={entropy.mean().item():.4f}")
        # --- DEBUG BLOCK END ---
    ema_scale = s.detach().cpu().numpy().mean()
    
    return loss, ema_scale

def critic_loss(critic_log, target_log, limit=20.0, buckets=255):
    # target_log: [Batch, 15]
    # critic_log: [Batch, 15, 255]
    
    target_sym = symlog(target_log)
    
    # Compute bucket width
    bucket_width = (limit * 2) / (buckets - 1)
    
    # Project target values into bucket space [0, buckets-1]
    buckets_cont = (target_sym + limit) / bucket_width

    # Find high-low bucket indices
    lower = buckets_cont.floor().long().clamp(0, buckets - 1)
    upper = buckets_cont.ceil().long().clamp(0, buckets - 1)

    # Compute weights for the two-hot encoding
    upper_w = buckets_cont - lower.float()
    lower_w = 1.0 - upper_w

    # Create the target distribution (same shape as critic_log)
    target_probs = torch.zeros_like(critic_log)

    # Apply weights using scatter_add
    # We need to unsqueeze indices to match the last dimension of target_probs
    target_probs.scatter_add_(-1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
    target_probs.scatter_add_(-1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))
    
    # Cross entropy between predicted logits and two-hot target distribution
    # Ensure critic_log is passed as logits for F.cross_entropy or use log_softmax
    loss = -torch.sum(target_probs * torch.log_softmax(critic_log, dim=-1), dim=-1).mean()

    return loss

