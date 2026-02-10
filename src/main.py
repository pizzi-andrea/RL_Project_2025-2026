import argparse
import os
from pathlib import Path
import torch
from globals import *
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from yaml import safe_load
from doomData import *
from dda import DynamicDataAllocator
from env_adapter import EnviromentAdapter
from network.controller.Policy import DreamerAgent
from network.encoder.VAE import VAE
from network.world.Dreamer import Dreamer, PERAdapter
from priority_functions import (
    AdversarialReplayPriority,
    CountBasedPriority,
    CuriosityReplayPriority,
)
import warnings

# Sopprime l'avviso di pkg_resources (Pygame)
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Sopprime l'avviso specifico del formato schermo di ViZDoom
warnings.filterwarnings("ignore", message="Detected screen format CRCGCB")

# Opzionale: sopprime tutti i messaggi di setuptools/pkg_resources a livello globale
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

torch.set_float32_matmul_precision('high')
def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--load_weights", action="store_true", required=False, help="load weights from src/weights directory")
    args = cmd.parse_args()
    
    with open("config.yaml", "r") as file:
        config = safe_load(file)
    
    image_processor = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    log_experiment = Path(config["global_config"]["Project_logs"])
    PER4train = AdversarialReplayPriority(config["PER_config"]["buffer_size"])

    # store inital trajectory in PER buffer
    for bucket in tqdm(Path(config["rssm_config"]["rssm_dataset_path_train"]).iterdir(), desc="load PER buffer"):
        for episode in bucket.iterdir():
            PER4train.store(episode)
    print(f"**load {len(PER4train)} items in PER buffer**")
    
    vae_trainer =  Trainer(
        logger=TensorBoardLogger(log_experiment, name="VAE"), 
        max_epochs=config["vae_config"]["train_epoch"]
    )
    rssm_trainer = Trainer(
        
                logger=TensorBoardLogger(log_experiment, name="RSSM"),
                callbacks=[PERAdapter(PER4train)],
                max_epochs=config["rssm_config"]["train_epoch"],
                limit_val_batches=1,
                log_every_n_steps=10,
    )

    policy_trainer = Trainer(
        logger=TensorBoardLogger(log_experiment, name="Policy"),
        max_epochs=config["policy_config"]["train_epoch"],
        log_every_n_steps=1,
    )
    
    
    if args.load_weights and Path(config["global_config"]["vae_w"]).exists():
        print(f"found VAE weights in {config["global_config"]["vae_w"]}")
        vae_model = VAE.load_from_checkpoint(config["global_config"]["vae_w"])
    else:
        vae_train_loader = DataLoader(
            DoomDataset(config["vae_config"]["VAE_dataset_path_train"], 
                        resize=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])), 
                        batch_size=config["vae_config"]["train_batch"], 
                        shuffle=False, 
                        num_workers=os.cpu_count()-1,
                        persistent_workers=True)
        vae_valid_loader = DataLoader(DoomDataset(config["vae_config"]["VAE_dataset_path_valid"], resize=(config["global_config"]["img_size_h"],config["global_config"]["img_size_l"])), batch_size=config["vae_config"]["train_batch"], shuffle=False, num_workers=os.cpu_count()-1, persistent_workers=True,)
    
        vae_model = VAE(latent_dim=config["vae_config"]["latent_dim"])
        vae_trainer.fit(vae_model, vae_train_loader, vae_valid_loader)
    
    if args.load_weights and Path(config["global_config"]["dreamer_w"]).exists():
        print(f"found Dreamer weights in {config["global_config"]["dreamer_w"]}")
        rssm_model = Dreamer.load_from_checkpoint(config["global_config"]["dreamer_w"], vae=vae_model)
    else:

        rssm_model = Dreamer(
                action_dim=config["global_config"]["action_shape"], 
                att_dim=config["global_config"]["att_shape"], 
                decode_shape=(config["global_config"]["img_channels"], config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                lr=config["rssm_config"]["lr"],
                beta_dyn=config["rssm_config"]["beta_dyn"],
                beta_pred=config["rssm_config"]["beta_pred"],
                beta_rep=config["rssm_config"]["beta_rep"],
                vae=vae_model,
                free_nats=config["rssm_config"]["free_nats"]  
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
                    lr=config["policy_config"]["lr"],
                    entropy_coef=config["policy_config"]["actor_entropy"]
        )
    
    envs = EnviromentAdapter(agent_model, DynamicDataAllocator(config["PER_config"]["cache"], PER4train))

    for epoch in range(1, config["global_config"]["dreamer_iterations"] + 1):
        
        # -- dreamer train --
        sampled_episods, priority_idxs, buffer_idxs, weights = collect_episods_from_buffer(PER4train, min(len(PER4train), min(len(PER4train), 300)))
        print("**rank_10 [pre] sample_episods**") 
        print(f"IDX: {buffer_idxs[:10]}")
        print(f"weights: {weights[:10]}")
        print(f"min/max weights: {min(weights)}/{max(weights)}")
        print(f"priority_nodes: {priority_idxs[:10]}")
        
        ## -- 
        rssm_train_loader = DataLoader(EpisodicDoomDataset(
                episods=sampled_episods,
                per_indexs=priority_idxs,
                obs_size=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                att_name=DoomNames.att_player_names,
                action_name=DoomNames.buttons_names,
                per_weights=weights,
                preprocessor=image_processor,
                chunk_size=config["global_config"]["chunk"]
            ),
            batch_size=config["rssm_config"]["train_batch"], 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=config["global_config"]["multi_cpu"], 
            pin_memory=True
        )
        rssm_trainer.fit_loop.max_epochs = epoch * config["rssm_config"]["train_epoch"]
        rssm_trainer.fit(rssm_model, train_dataloaders=rssm_train_loader, val_dataloaders=rssm_train_loader)

        # -- policy train --
        sampled_episods, priority_idxs, buffer_idxs, weights = collect_episods_from_buffer(PER4train, min(len(PER4train), 300))
        print("**rank_10 [post] sample_episods**") 
        print(f"IDX: {buffer_idxs[:10]}")
        print(f"weights: {weights[:10]}")
        print(f"min/max weights: {min(weights)}/{max(weights)}")
        print(f"priority_nodes: {priority_idxs[:10]}")
        
        
        policy_train_loader = DataLoader(EpisodicDoomDataset(
            
                episods=sampled_episods,
                per_indexs=priority_idxs,
                obs_size=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                att_name=DoomNames.att_player_names,
                action_name=DoomNames.buttons_names,
                per_weights=weights,
                chunk_size=config["global_config"]["chunk"],
                preprocessor=image_processor,
                
            ), 
            batch_size=config["policy_config"]["train_batch"], 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=config["global_config"]["multi_cpu"], 
            pin_memory=True
        )
        policy_trainer.fit_loop.max_epochs = epoch * config["policy_config"]["train_epoch"]
        policy_trainer.fit(agent_model,
                train_dataloaders=policy_train_loader,
        )

        
        if (epoch > config["global_config"]["warmup"]):
            print("Collect new trajectorys ...")
            for env, n in zip(config["global_config"]["envs"], config["global_config"]["envs_episods"]):
                reword = envs.act_in_env(env, n_episodes=n, frame_skip=4, max_length=config["global_config"]["max_episode_length"])
                print(f"mean reword collected by policy {reword} in env {env}")
        else:
            print("warmup ...")
        




if __name__ == '__main__':
    main()