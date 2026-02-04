import argparse
import os
from pathlib import Path

import gymnasium
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from yaml import safe_load
from vizdoom import gymnasium_wrapper

from doomData import DoomDataset, EpisodicDoomDataset, collate_fn
from network.controller.Policy import DreamerAgent
from network.encoder.VAE import VAE
from network.world.Dreamer import Dreamer, PERAdapter
from priority_functions import (
    AdversarialReplayPriority,
    CountBasedPriority,
    CuriosityReplayPriority,
)
from imaginated_rollout import save_imagined_trajectory

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
        batch_size=config["rssm_config"]["train_batch"], shuffle=True, collate_fn=collate_fn, num_workers=8)
    
    policy_train_loader = DataLoader(EpisodicDoomDataset(
        config["policy_config"]["policy_dataset_path_train"], 
        max_length=config["policy_config"]["burn_in"], 
        priority_buffer=PER4train, 
        preprocessor=Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])), 
        batch_size=config["policy_config"]["train_batch"] , shuffle=True, collate_fn=collate_fn, num_workers=8)
    
    policy_valid_loader = DataLoader(EpisodicDoomDataset(
        config["policy_config"]["policy_dataset_path_train"], 
        max_length=config["policy_config"]["burn_in"], 
        priority_buffer=PER4train, 
        preprocessor=Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])), 
        batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8)
    


    
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

    policy_trainer = Trainer(
        logger=TensorBoardLogger(log_experiment.joinpath("Policy")),
        max_epochs=config["policy_config"]["train_epoch"],
        log_every_n_steps=50
    )
    
    
  
    
    vae_trainer.fit(VAE(latent_dim=config["vae_config"]["latent_dim"]), vae_train_loader, vae_valid_loader)
    
    for epoch in range(1, config["global_config"]["dreamer_iterations"]):
        print(f"Global iteration ({epoch})")
        rssm_trainer.fit(Dreamer(
            action_dim=config["global_config"]["action_shape"], 
            att_dim=config["global_config"]["att_shape"], 
            image_channels=config["global_config"]["img_channels"],
            vae=vae_trainer.model
        ),  rssm_train_loader)
        
        policy_trainer.fit(
            DreamerAgent(
                rssm_trainer.model, 
                action_space_dim=config["global_config"]["action_shape"],
                horizont=config["policy_config"]["horizont"],
                burn_in=config["policy_config"]["burn_in"]
                ),
                train_dataloaders=policy_train_loader,
                val_dataloaders=policy_valid_loader
        )
        save_imagined_trajectory(
            env_id="VizdoomBasic-v0",
            agent=policy_trainer.model.to(policy_trainer.model.device),
            out_dir=str(log_experiment.joinpath("Policy", "imagination")),
            burn_in_steps=config["policy_config"]["burn_in"],
            imagine_horizon=config["policy_config"]["horizont"],
            seed=0,
            tag=epoch,
        )


    

    
if __name__ == '__main__':
    main()