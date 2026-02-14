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


def symexp(x):
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1)

@torch.no_grad()
def twohot(x, B):
    """
    x: Target scalari (es. lambda-returns), shape [...]
    B: Centri dei bin (già in scala symexp), shape [N]
    returns: Vettore two-hot, shape [..., N]
    """
   
    k = torch.sum(B < x.unsqueeze(-1), dim=-1) # Shape [...]
    
    k = torch.clamp(k, 1, B.size(0) - 1)
    
    # 3. Recupera i valori dei bin b_k e b_{k+1}
    b_k = B[k - 1]
    b_k1 = B[k]
    
    # 
    # w_k = |b_{k+1} - x| / |b_{k+1} - b_k|
    # w_k1 = |b_k - x| / |b_{k+1} - b_k|
    denom = b_k1 - b_k
    w_k = (b_k1 - x) / denom
    w_k1 = (x - b_k) / denom
    
    # 5. Costruzione del vettore di output
    out = torch.zeros(*x.shape, B.size(0), device=x.device)
    
    # Usiamo scatter per posizionare i pesi negli indici corretti
    # out.scatter_(dim, index, value)
    out.scatter_(-1, (k - 1).unsqueeze(-1), w_k.unsqueeze(-1))
    out.scatter_(-1, k.unsqueeze(-1), w_k1.unsqueeze(-1))
    
    # 6. Clipping finale dei pesi tra 0 e 1 (per valori di x fuori dal range dei bin)
    return torch.clamp(out, 0, 1)

@torch.no_grad() 
def lambda_returns(rewards, values, continues, gamma=0.997, lam=0.95):
    """
    
    """

    #
    if values.shape[1] == rewards.shape[1]:
        
        
        rewards = rewards[:, :-1]
        continues = continues[:, :-1]
        
        v_pred = values[:, :-1]
        next_val = values[:, 1:]
    else:
        
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




def actor_loss(td_return, critic_values, action_log_prob, entropy, ema_scaler, continues, entropy_scale=3e-4, debug=False):
    advantages = td_return - critic_values
    s = ema_scaler(td_return)
    scaled_advantages = (advantages / torch.clamp(s, min=1.0)).detach()

    loss = -(scaled_advantages*action_log_prob + entropy_scale*entropy) * continues

    # scale loss respect only valid steps
    total_valid_steps = continues.sum()
    final_loss = loss.sum() / torch.clamp(total_valid_steps, min=1.0)

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
    
    return final_loss, ema_scale

def critic_loss(v_hat, target, continues):

    log_probs = torch.nn.functional.log_softmax(v_hat, dim=-1)   
    per_step_loss = -(target * log_probs).sum(dim=-1)    
    masked_loss = per_step_loss * continues
    total_valid_steps = continues.sum()
    final_loss = masked_loss.sum() / torch.clamp(total_valid_steps, min=1.0)

    return final_loss


