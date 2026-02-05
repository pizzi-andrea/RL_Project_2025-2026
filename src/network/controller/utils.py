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
    def __init__(self, decay=0.99, eps=1e-6):
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

@no_grad
def lambda_returns(rewards, values, continues, gamma=0.997, lam=0.95):
    """
    Computes Lambda-returns using GAE (General Advantage Estimation).
    
    Args:
        rewards:   [Batch, Horizon]    (e.g., length 15 or 16)
        values:    [Batch, Horizon+1]  (e.g., length 16: steps 0..14 + step 15 bootstrap)
        continues: [Batch, Horizon]    (e.g., discount/non-terminal flags for each step)
    
    Returns:
        lambda_returns: [Batch, Horizon] (Length 15, aligned with Action LogProbs)
    """
    
    # Determine the effective horizon based on the provided Values.
    # If values has 16 steps, we calculate returns for the first 15 steps.
    batch_size, T_plus_1 = values.shape
    horizon = T_plus_1 - 1 # If T_plus_1 is 16, horizon is 15
    
    # Slice rewards and continues to match the calculation horizon
    # (Extract the first 15 steps)
    rewards = rewards[:, :horizon]
    continues = continues[:, :horizon]
    
    # Initialize Advantage buffer
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    
    # Backward Pass: t iterates from horizon-1 down to 0 (e.g., 14 to 0)
    for t in reversed(range(horizon)):
        # 1. Retrieve "Next Value" directly from the values array
        # Since values has length 16, values[:, t+1] is always valid
        # (For t=14, this accesses the bootstrap index 15)
        next_val = values[:, t+1]
        
        # 2. Retrieve current state value
        curr_val = values[:, t]
        
        # 3. Continuation mask (1.0 - done)
        # continues[:, t] indicates if the transition from t to t+1 is valid
        next_non_terminal = continues[:, t]

        # 4. Compute TD Error (delta)
        # delta = r_t + gamma * v_{t+1} * mask - v_t
        delta = rewards[:, t] + gamma * next_val * next_non_terminal - curr_val
        
        # 5. Recursive GAE Update
        # A_t = delta + (gamma * lambda * mask) * A_{t+1}
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
        # Store the calculated advantage
        advantages[:, t] = last_gae_lam
        
    # 6. Final Calculation: Lambda Returns = Advantages + Values
    # R_t = A_t + V_t
    targets = advantages + values[:, :horizon]
    
    # Detach from graph as these are targets for the Critic/Actor
    return targets.detach()





def make_video(frames, output_path, fps=30):
    frames = frames.cpu().detach()
    frames_video = (sigmoid(frames)*255).to(dtype=torch.uint8)
    frames_video = frames_video.permute(0,2,3,1).numpy()
    video = frames_video[:, :, :, ::-1]
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

    return loss

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

