import argparse
import os
from trainer import VAE_Trainer, DreamerXP_Trainer
from network.encoder.VAE import VAE
from priority_functions import AdversarialReplayPriority
from pathlib import Path
from lightning import Trainer
import numpy as np
import torch
from torch import no_grad
from env_info import ATT_PLAYER
from priority_functions import CuriosityReplayPriority, CountBasedPriority, AdversarialReplayPriority
from doomData import DoomDataset, EpisodicDoomDataset, collate_fn
from doomData import Transition, Episode
from network.encoder.VAE import VAE
from network.RSSM.rssm import RSSM
from network.world.Dreamer import Dreamer, PERAdapter
from network.controller.Policy import Policy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer as LTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from network.controller.Policy import DreamerAgent
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
import gymnasium
from vizdoom import gymnasium_wrapper

torch.set_float32_matmul_precision('medium')
def main():

    log_experiment = Path("Experiment")
    PER4train = AdversarialReplayPriority(50)

    vae_train_loader = DataLoader(DoomDataset("Doom4VAE/train", resize=(240,320)), batch_size=32,     shuffle=True, num_workers=os.cpu_count()-1)
    vae_valid_loader = DataLoader(DoomDataset("Doom4VAE/validation", resize=(240,320)), batch_size=32, shuffle=False, num_workers=os.cpu_count()-1)
    
    rssm_train_loader = DataLoader(EpisodicDoomDataset("Doom4RSSM/train", max_length=300, priority_buffer=PER4train, preprocessor=Compose(
        [ToImage(), ToDtype(dtype=torch.float32, scale=True)]
    )), batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=8)

    
    vae_trainer =  Trainer(logger=TensorBoardLogger(log_experiment.joinpath("VAE")), max_epochs=10)
    rssm_trainer = Trainer(
        
                logger=TensorBoardLogger(log_experiment.joinpath("RSSM")),
                callbacks=[PERAdapter(PER4train)],
                max_epochs=5,
                log_every_n_steps=10)
    
    
  
  
    vae_trainer.fit(VAE(latent_dim=128), vae_train_loader, vae_valid_loader)
    rssm_trainer.fit(Dreamer(action_dim=16, att_dim=6, image_channels=3,vae=vae_trainer.model), rssm_train_loader)

    
if __name__ == '__main__':
    main()