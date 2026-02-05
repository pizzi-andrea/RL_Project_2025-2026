import argparse
import os
from pathlib import Path

import gymnasium
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToDtype, ToImage, Resize
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

torch.set_float32_matmul_precision('medium')
def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--load_weights", action="store_true", required=False, help="load weights from src/weights directory")
    args = cmd.parse_args()

    with open("config.yaml", "r") as file:
        config = safe_load(file)
    
    
    log_experiment = Path(config["global_config"]["Project_logs"])
    PER4train = AdversarialReplayPriority(config["PER_config"]["buffer_size"])

    
    rssm_train_loader = DataLoader(EpisodicDoomDataset(
        config["rssm_config"]["rssm_dataset_path_train"], 
        max_length=config["rssm_config"]["max_episode_length"], 
        priority_buffer=PER4train, 
        preprocessor=Compose([ToImage(), Resize((config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])), ToDtype(dtype=torch.float32, scale=True)])), 
        batch_size=config["rssm_config"]["train_batch"], shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count()-1, pin_memory=True)
    
    
    #rssm_dataset_path_valid
    policy_train_loader = DataLoader(EpisodicDoomDataset(
        config["policy_config"]["policy_dataset_path_train"], 
        max_length=config["policy_config"]["burn_in"], 
        priority_buffer=PER4train, 
        preprocessor=Compose([ToImage(), Resize((120, 160)),ToDtype(dtype=torch.float32, scale=True)])), 
        batch_size=config["policy_config"]["train_batch"] , shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count()-1)

    vae_trainer =  Trainer(
        logger=TensorBoardLogger(log_experiment.joinpath("VAE")), 
        max_epochs=config["vae_config"]["train_epoch"]
    )
    rssm_trainer = Trainer(
        
                logger=TensorBoardLogger(log_experiment.joinpath("RSSM")),
                callbacks=[PERAdapter(PER4train)],
                max_epochs=config["rssm_config"]["train_epoch"],
                log_every_n_steps=50,
                limit_val_batches=1
    )

    policy_trainer = Trainer(
        logger=TensorBoardLogger(log_experiment.joinpath("Policy")),
        max_epochs=config["policy_config"]["train_epoch"],
        log_every_n_steps=50,
        
    )
    
    if args.load_weights and Path(config["global_config"]["vae_w"]).exists():
        print(f"found VAE weights in {config["global_config"]["vae_w"]}")
        vae_model = VAE.load_from_checkpoint(config["global_config"]["vae_w"])
    else:
        vae_train_loader = DataLoader(DoomDataset(config["vae_config"]["VAE_dataset_path_train"], resize=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])), batch_size=config["vae_config"]["train_batch"], shuffle=True, num_workers=os.cpu_count()-1)
        vae_valid_loader = DataLoader(DoomDataset(config["vae_config"]["VAE_dataset_path_valid"], resize=(config["global_config"]["img_size_h"],config["global_config"]["img_size_l"])), batch_size=config["vae_config"]["train_batch"], shuffle=False, num_workers=os.cpu_count()-1)
    
        vae_model = VAE(latent_dim=config["vae_config"]["latent_dim"])
        vae_trainer.fit(vae_model, vae_train_loader, vae_valid_loader)
    
    if args.load_weights and Path(config["global_config"]["dreamer_w"]).exists():
        print(f"found Dreamer weights in {config["global_config"]["dreamer_w"]}")
        rssm_model = Dreamer.load_from_checkpoint(config["global_config"]["dreamer_w"])
    else:

        rssm_model = Dreamer(
                action_dim=config["global_config"]["action_shape"], 
                att_dim=config["global_config"]["att_shape"], 
                decode_shape=(config["global_config"]["img_channels"], config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                lr=config["rssm_config"]["lr"],
                beta_dyn=config["rssm_config"]["beta_dyn"],
                beta_pred=config["rssm_config"]["beta_pred"],
                beta_rep=config["rssm_config"]["beta_rep"],
                vae=vae_model
        )  

    if args.load_weights and Path(config["global_config"]["policy_w"]).exists():
        print(f"found Policy weights in {config["global_config"]["policy_w"]}")
        agent_model = DreamerAgent.load_from_checkpoint(config["global_config"]["policy_w"], dreamer_model=rssm_model)
    else:
        agent_model = DreamerAgent(
                    rssm_model, 
                    action_space_dim=config["global_config"]["action_shape"],
                    horizont=config["policy_config"]["horizont"],
                    burn_in=config["policy_config"]["burn_in"],
                    lr=config["policy_config"]["lr"]
        )
    
    
    
    for epoch in range(1, config["global_config"]["dreamer_iterations"]):
        print(f"Global iteration ({epoch})")
        
        rssm_trainer.fit(rssm_model, train_dataloaders=rssm_train_loader, val_dataloaders=rssm_train_loader)
        
        policy_trainer.fit(agent_model,
                train_dataloaders=policy_train_loader,
                
        )

    

    
if __name__ == '__main__':
    main()