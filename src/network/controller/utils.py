import cv2 as cv
import torch 
from torch import nn
from torch.nn.functional import sigmoid
class EMAScalar(nn.Module):
    def __init__(self, decay=0.98, eps=1e-6):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, values):
        values = values.detach()
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
    #x = x.clamp(mm, mx)
    return torch.sign(x)*(torch.exp(torch.abs(x)) - 1)


# utils.py modificato
@torch.no_grad()
def lambda_returns(rewards, expected_values, continues, gamma=0.997, lam=0.95):
    """
    Tutti gli input devono essere scalari [B, T] nello spazio REALE.
    """
    returns = torch.zeros_like(expected_values)
    # Il valore di bootstrap dell'ultimo step non deve essere perso
    last_ret = expected_values[:, -1]
    returns[:, -1] = last_ret 
    
    for t in reversed(range(rewards.shape[1] - 1)):
        # V_{t} = r_t + \gamma * c_t * [ (1-\lambda) V_{t+1} + \lambda R_{t+1} ]
        ret = rewards[:, t] + gamma * continues[:, t] * ((1 - lam) * expected_values[:, t+1] + lam * last_ret)
        returns[:, t] = ret
        last_ret = ret
        
    return returns


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


def actor_loss(td_return, critic_values, action_log_prob, entropy, ema_scaler, continues, entropy_scale=3e-4):
    """Actor loss REINFORCE style"""
    s = ema_scaler(td_return)
  
    normalized_advantage = ((td_return - critic_values) / torch.clamp(s, min=1)).detach()
    
    per_step_objective = (normalized_advantage * action_log_prob) + (entropy_scale * entropy)
    masked_objective = per_step_objective * continues
    lengths = continues.sum(dim=1).clamp(min=1.0)
    final_loss = -(masked_objective.sum(dim=1) / lengths).mean()
    
    
    ema_scale = s.detach().cpu().numpy().mean()
    avantage = ((normalized_advantage * action_log_prob).detach().sum() / lengths).mean().item()
    entropy = ((entropy_scale * entropy).detach().sum() / lengths).mean().item()
    return final_loss, ema_scale, avantage, entropy

def critic_loss(v_hat, target, continues):
    
    log_probs = torch.nn.functional.log_softmax(v_hat, dim=-1)   
    
    per_step_loss = -(target * log_probs).sum(dim=-1)
    
    lengths = continues.sum().clamp(min=1.0)
    final_loss = (per_step_loss * continues).sum() / lengths

    return final_loss

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
    min_b = B[0]
    max_b = B[-1]
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