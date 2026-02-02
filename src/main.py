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
from yaml import safe_load
import gymnasium
from vizdoom import gymnasium_wrapper

torch.set_float32_matmul_precision('medium')
def main():
    with open("config.yaml", "r") as file:
        config = safe_load(file)
    

    log_experiment = Path(config["global_config"]["Project_logs"])
    PER4train = AdversarialReplayPriority(config["PER_config"]["buffer_size"])

    vae_train_loader = DataLoader(DoomDataset(config["vae_config"]["VAE_dataset_path_train"], resize=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])), batch_size=config["vae_config"]["train_batch"], shuffle=True, num_workers=os.cpu_count()-1)
    vae_valid_loader = DataLoader(DoomDataset(config["vae_config"]["VAE_dataset_path_valid"], resize=(config["global_config"]["img_size_h"],config["global_config"]["img_size_l"])), batch_size=config["vae_config"]["train_batch"], shuffle=False, num_workers=os.cpu_count()-1)
    
    rssm_train_loader = DataLoader(EpisodicDoomDataset(
        config["rssm_config"]["rssm_dataset_path_train"], 
        max_length=config["rssm_config"]["max_episode_length"], 
        priority_buffer=PER4train, 
        preprocessor=Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])), 
        batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=8)

    
    vae_trainer =  Trainer(
        logger=TensorBoardLogger(log_experiment.joinpath("VAE")), 
        max_epochs=config["vae_config"]["train_epoch"]
    )
    rssm_trainer = Trainer(
        
                logger=TensorBoardLogger(log_experiment.joinpath("RSSM")),
                callbacks=[PERAdapter(PER4train)],
                max_epochs=config["rssm_config"]["train_epoch"],
                log_every_n_steps=10
    )
    
    
  
  
    vae_trainer.fit(VAE(latent_dim=config["vae_config"]["latent_dim"]), vae_train_loader, vae_valid_loader)
    rssm_trainer.fit(Dreamer(
        action_dim=config["global_config"]["action_shape"], 
        att_dim=config["global_config"]["att_shape"], 
        image_channels=config["global_config"]["img_channels"],
        vae=vae_trainer.model
    ),  rssm_train_loader)

    
if __name__ == '__main__':
    main()