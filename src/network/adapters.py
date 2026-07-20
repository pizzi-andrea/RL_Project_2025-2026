import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np

class Adapter(nn.Module):
    """High level class for adapters class"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
class PERAdapter(L.Callback):
    """Unified interface class for PER"""
    def __init__(self, buffer):
        super().__init__()
        self.buffer_train = buffer
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        index = outputs["episodes_idx"]
        loss = outputs["episodes_loss"]
        
        self.buffer_train.update(priority_idx=index, losses=loss)
        
        if batch_idx == 0:     
            self.buffer_train.info()
           
class EncoderAdapter(Adapter):
    """
    Unified interface for encoder module
    """
    def __init__(self, encoder):
        super().__init__(encoder)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] oppure [B,C,H,W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            
            x = x.reshape(B * T, C, H, W)
            
            feat = self.encoder.encode(x)  

            return feat.view(B, T, -1)
        else:
            feat = self.encoder.encode(x)
           
            return feat

class DecoderAdapter(Adapter):
    """
    Unified interface for decoder module
    """
    def __init__(self, decoder_module):

        super().__init__(decoder_module)
        self.res = decoder_module.input_res # (64, 64)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 3:
            B, T, D = feat.shape
   
            feat_flat = feat.reshape(B * T, D) 
            
      
            imgs = self.encoder.decode(feat_flat) 
    
            return imgs.view(B, T, 3, self.res[0], self.res[1])
        else:
          
            return self.encoder.decode(feat)