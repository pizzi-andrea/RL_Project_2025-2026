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
        """Exponential Mobile Mean used to stabilize Critic learnig"""
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

        # == prevent peaks and noise values
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
        """symlog operation presented in DreaamerV3 paper"""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
        
def symexp(x):
    """Inverse of symlog operator"""
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1.0)
@torch.no_grad()
def lambda_returns(rewards, expected_values, execution_mask, gamma=0.997, lam=0.95):
    """
    
    """
    returns = torch.zeros_like(expected_values)
    
    last_ret = expected_values[:, -1]
    returns[:, -1] = last_ret 
    
    for t in reversed(range(expected_values.shape[1] - 1)):
        target = (1 - lam) * expected_values[:, t + 1] + lam * last_ret
        returns[:, t] = rewards[:, t] + gamma * execution_mask[:, t] * target
        last_ret = returns[:, t]
        
    return returns[:, :-1]





def make_video(frames, output_path, fps=30):
    frames = frames.cpu().detach()
    frames_video = (frames*255).to(dtype=torch.uint8)
    frames_video = frames_video.permute(0,2,3,1).numpy()
    frames_video = frames_video[:, :, :, ::-1]
    num_frames, height, width, _ = frames_video.shape
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        video_writer.write(frames_video[i])
    
    video_writer.release()
    print(f"new video saved in : {output_path}")

def _unitwise_norm(x):
    if x.ndim <= 1:
        return x.norm(2)
    return x.norm(2, dim=tuple(range(1, x.ndim)), keepdim=True)

def AGC(parameters, clip_factor=0.01, eps=1e-3):
    for p in parameters:
        if p.grad is None or p.ndim < 2:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        
        p_norm = _unitwise_norm(p_data).clamp_(min=eps)
        g_norm = _unitwise_norm(g_data)
        
        max_norm = p_norm * clip_factor
        trigger = g_norm > max_norm
        
        scaled_g = g_data * (max_norm / g_norm.clamp(min=eps))
        p.grad.copy_(torch.where(trigger, scaled_g, g_data))




@torch.no_grad()
def twohot(x, B):
    min_b = B[0]
    max_b = B[-1]
    x = x.clamp(min_b, max_b)
    num_buckets = B.shape[0]
    step = (max_b - min_b) / (num_buckets - 1)

    idx = (x - min_b) / step
    idx = idx.clamp(0, num_buckets - 1 - 1e-5)
    
    lower = idx.floor().long().clamp(0, num_buckets - 2)
    upper = (lower + 1).clamp(0, num_buckets - 1)
    
    upper_weight = (idx - lower.float()).clamp(0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    
    out = torch.zeros(*x.shape, num_buckets, device=x.device)
    
    out.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    out.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    
    return out


def force_list(elements: Optional[Any] = None, to_tuple: bool = False) -> Union[List, Tuple]:
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