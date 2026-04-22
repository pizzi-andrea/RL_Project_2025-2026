import numpy as np
import cv2 as cv

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
from utils import *
from dda import DynamicDataAllocator
from env_adapter import EnviromentAdapter
from network.controller.Policy import DreamerAgent
from network.encoder.VAE import VAE
from network.worldModel.dreamer import Dreamer
from network.adapters import PERAdapter
from torchvision.transforms.v2 import *
from torchvision.transforms import InterpolationMode
from priority_functions import (
    AdversarialReplayPriority,
    CountBasedPriority,
    CuriosityReplayPriority,
)
import warnings
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import RichProgressBar
class StepLimitCallback(Callback):
    def __init__(self, steps_per_round):
        super().__init__()
        self.steps_per_round = steps_per_round
        self.batch_count = 0

    def on_train_start(self, trainer, pl_module):
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        if self.batch_count >= self.steps_per_round:
            trainer.should_stop = True

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", message="Detected screen format CRCGCB")

def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--load_weights", action="store_true", required=False, help="load weights from src/weights directory")
    args = cmd.parse_args()
    with open("config.yaml", "r") as file:
        config = safe_load(file)
    validation_log = Path(config["global_config"]["Project_logs"]).joinpath("best_epochs")
    validation_log.mkdir(parents=True, exist_ok=True)
    obs_size = (config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])
    def robust_preprocessor(x):
        
        if len(x.shape) == 3:
            x = x[np.newaxis,:,:,:]
        
        t = []
        for item in x:
            lab = cv.cvtColor(item, cv.COLOR_RGB2LAB)
            l, a, b = cv.split(lab)

    
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
        
            limg = cv.merge((cl, a, b))
            final_img = cv.cvtColor(limg, cv.COLOR_LAB2RGB)
            t.append(final_img)
        
        t = np.stack(t, axis=0)
        t = torch.from_numpy(t).permute(0, 3, 1, 2).contiguous()
    
    # dataset preprocess
    def simple_preprocessor(x):
        if len(x.shape) == 3:
            x = x[np.newaxis,:,:,:]
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        return x
    
    
    def simple_post_processor(x):
        x = x[100:, 54:267]
        x = cv.resize(x, obs_size, interpolation=cv.INTER_NEAREST)
        return x

    def simple_preprocessor_float32(x):
        x = x[100:, 54:267]
        x = cv.resize(x, obs_size, interpolation=cv.INTER_NEAREST)
        if len(x.shape) == 3:
            x = x[np.newaxis,:,:,:]

############################################################
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        x = x.float().div_(255.0).detach()
        return x
    

  
    image_processor = simple_preprocessor
    log_experiment = Path(config["global_config"]["Project_logs"])
    PER4train = CuriosityReplayPriority(
        config["PER_config"]["buffer_size"], 
        
        alpha=config["curiosity_config"]["alpha"],
        beta=config["curiosity_config"]["beta"],
        eps=config["curiosity_config"]["eps"],
        p_max=config["curiosity_config"]["p_max"],
        c=config["curiosity_config"]["c"]
        )

    for bucket in tqdm(Path(config["rssm_config"]["rssm_dataset_path_train"]).iterdir(), desc="load PER buffer"):
        for episode in bucket.iterdir():
            PER4train.store(episode)
    print(f"**load {len(PER4train)} items in PER buffer**")
    
    
    
    rssm_trainer = Trainer(
        logger=TensorBoardLogger(log_experiment, name="RSSM"),
        callbacks=[PERAdapter(PER4train), StepLimitCallback(config["rssm_config"]["train_steps"]), RichProgressBar(leave=True)],
        max_epochs=-1,
        log_every_n_steps=1,
        accumulate_grad_batches=config["rssm_config"]["batch_acc"],
        precision=config["global_config"]["precision"]
    )

    policy_trainer = Trainer(
        logger=TensorBoardLogger(log_experiment, name="Policy"),
        callbacks=[StepLimitCallback(config["policy_config"]["train_steps"]), RichProgressBar(leave=True)],
        max_epochs=-1,
        log_every_n_steps=1,
        accumulate_grad_batches=config["policy_config"]["batch_acc"],
        precision=config["global_config"]["precision"],
        gradient_clip_val=100.0,
        gradient_clip_algorithm="norm"
        
    )
    
    #if args.load_weights and Path(config["global_config"]["vae_w"]).exists():
        #vae_model = VAE.load_from_checkpoint(config["global_config"]["vae_w"])
    #else:
    vae_model = VAE(latent_dim=config["vae_config"]["latent_dim"], model_dim=config["vae_config"]["model_dim"])
        #if config["vae_config"]["train_epoch"] > 0:
        #    vae_trainer =  Trainer(
        #        logger=TensorBoardLogger(log_experiment, name="VAE"), 
        #        max_epochs=config["vae_config"]["train_epoch"]
        #    )
        #    vae_train_loader = DataLoader(
        #        DoomDataset(config["vae_config"]["VAE_dataset_path_train"], 
        #                    resize=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])), 
        #                    batch_size=config["vae_config"]["train_batch"], 
        #                    shuffle=False, 
        #                    num_workers=os.cpu_count()-1,
        #                    persistent_workers=True)
        #    vae_valid_loader = DataLoader(DoomDataset(config["vae_config"]["VAE_dataset_path_valid"], resize=(config["global_config"]["img_size_h"],config["global_config"]["img_size_l"])), batch_size=config["vae_config"]["train_batch"], shuffle=False, num_workers=os.cpu_count()-1, persistent_workers=True,)
        #    vae_trainer.fit(vae_model, vae_train_loader, vae_valid_loader)
        
    if args.load_weights and Path(config["global_config"]["dreamer_w"]).exists():
        rssm_model = Dreamer.load_from_checkpoint(config["global_config"]["dreamer_w"],
                action_dim          =   config["global_config"]["action_shape"], 
                att_dim             =   config["global_config"]["att_shape"], 
                decode_shape        =   (config["global_config"]["img_channels"], config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                lr                  =   config["rssm_config"]["lr"],
                beta_dyn            =   config["rssm_config"]["beta_dyn"],
                beta_pred           =   config["rssm_config"]["beta_pred"],
                beta_rep            =   config["rssm_config"]["beta_rep"],
                #vae                 =   vae_model,
                burn_in             =    config["rssm_config"]["burn_in"],
                free_nats           =   config["rssm_config"]["free_nats"],
                unimix              =   config["rssm_config"]["unimix"],
                internal_hidden_dim =   config["rssm_config"]["common_latent_dim"],
                stoch_classes       =   config["rssm_config"]["stochastic_classes"],
                stoch_latents       =   config["rssm_config"]["stochastic_latent"],
                rssm_hidden_dim     =   config["rssm_config"]["rssm_latent_dim"],
                h_dim               =   config["rssm_config"]["detterministic_latent_dim"],
                buckets             =   config["rssm_config"]["buckets"],
                min_buckets         =   config["rssm_config"]["min_buckets"],
                max_buckets         =   config["rssm_config"]["max_buckets"],
                alpha_obs           =   config["rssm_config"]["a_obs"],
                alpha_att           =   config["rssm_config"]["a_att"],
                alpha_reward        =   config["rssm_config"]["a_rew"],
                alpha_continue      =   config["rssm_config"]["a_cont"],

                #preprocess_callback=base_callback            
        )
    else:
        rssm_model = Dreamer(
                action_dim          =   config["global_config"]["action_shape"], 
                att_dim             =   config["global_config"]["att_shape"], 
                decode_shape        =   (config["global_config"]["img_channels"], config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                lr                  =   config["rssm_config"]["lr"],
                beta_dyn            =   config["rssm_config"]["beta_dyn"],
                beta_pred           =   config["rssm_config"]["beta_pred"],
                beta_rep            =   config["rssm_config"]["beta_rep"],
                vae                 =   vae_model,
                burn_in             =    config["rssm_config"]["burn_in"],
                free_nats           =   config["rssm_config"]["free_nats"],
                unimix              =   config["rssm_config"]["unimix"],
                internal_hidden_dim =   config["rssm_config"]["common_latent_dim"],
                stoch_classes       =   config["rssm_config"]["stochastic_classes"],
                stoch_latents       =   config["rssm_config"]["stochastic_latent"],
                rssm_hidden_dim     =   config["rssm_config"]["rssm_latent_dim"],
                h_dim               =   config["rssm_config"]["detterministic_latent_dim"],
                buckets             =   config["rssm_config"]["buckets"],
                min_buckets         =   config["rssm_config"]["min_buckets"],
                max_buckets         =   config["rssm_config"]["max_buckets"],
                alpha_obs           =   config["rssm_config"]["a_obs"],
                alpha_att           =   config["rssm_config"]["a_att"],
                alpha_reward        =   config["rssm_config"]["a_rew"],
                alpha_continue      =   config["rssm_config"]["a_cont"],
                
                #preprocess_callback=base_callback
        ) 
    
    if args.load_weights and Path(config["global_config"]["policy_w"]).exists():
        agent_model = DreamerAgent.load_from_checkpoint(config["global_config"]["policy_w"], 
                    dreamer_model       =   rssm_model, 
                    action_space_dim    =   config["global_config"]["action_shape"],
                    horizon             =   config["policy_config"]["horizon"],
                    burn_in             =   config["policy_config"]["burn_in"],
                    lr                  =   config["policy_config"]["lr"],
                    entropy_coef        =   config["policy_config"]["actor_entropy"],
                    buckets             =   config["policy_config"]["buckets"],
                    hidden_dim          =   config["policy_config"]["hidden_dim"],
                    unimix              =   config["policy_config"]["unimix"],
                    ema_decay           =   config["policy_config"]["ema_decay"],
                    ema_epsilon         =   config["policy_config"]["ema_epsilon"],
                    min_buckets         =   config["policy_config"]["min_buckets"],
                    max_buckets         =   config["policy_config"]["max_buckets"],
                    #preprocess_callback =   base_callback,
        
        )
    else:
        agent_model = DreamerAgent(
                    dreamer_model       =   rssm_model, 
                    action_space_dim    =   config["global_config"]["action_shape"],
                    horizon             =   config["policy_config"]["horizon"],
                    burn_in             =   config["policy_config"]["burn_in"],
                    lr                  =   config["policy_config"]["lr"],
                    entropy_coef        =   config["policy_config"]["actor_entropy"],
                    buckets             =   config["policy_config"]["buckets"],
                    hidden_dim          =   config["policy_config"]["hidden_dim"],
                    unimix              =   config["policy_config"]["unimix"],
                    ema_decay           =   config["policy_config"]["ema_decay"],
                    ema_epsilon         =   config["policy_config"]["ema_epsilon"],
                    min_buckets         =   config["policy_config"]["min_buckets"],
                    max_buckets         =   config["policy_config"]["max_buckets"],
                    #preprocess_callback=base_callback,
                    
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    envs = EnviromentAdapter(
        agent_model, 
        DynamicDataAllocator(config["PER_config"]["cache"], 
        PER4train), device=device
    )
    print(f"Total steps for DREAMER: {config['rssm_config']['sample_episodes']/config['rssm_config']['train_batch']}")
    print(f"Total steps for Policy: {config['policy_config']['sample_episodes']/config['policy_config']['train_batch']}")
    for epoch in range(1, config["global_config"]["dreamer_iterations"] + 1):
        print(f"Start epoch iteration {epoch}/{config['global_config']['dreamer_iterations']}")
        sampled_episodes, priority_idxs, _, weights = collect_episodes_from_buffer(PER4train, min(len(PER4train), config["rssm_config"]["sample_episodes"]))
        PER4train.info()
        
        rssm_train_loader = DataLoader(EpisodicDoomDataset(
                episodes=sampled_episodes,
                per_indexs=priority_idxs,
                obs_size=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
                att_name=DoomNames.att_player_names,
                action_name=DoomNames.buttons_names,
                per_weights=weights,
                preprocessor=image_processor,
                chunk_size=config["global_config"]["chunk"],
                
            ),
            batch_size=config["rssm_config"]["train_batch"], 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=config["global_config"]["multi_cpu"], 
            pin_memory=True
        )

        rssm_trainer.fit(rssm_model, train_dataloaders=rssm_train_loader)
        
        sampled_episodes, priority_idxs, _, weights = collect_episodes_from_buffer(PER4train, min(len(PER4train), config["policy_config"]["sample_episodes"]))
        PER4train.info()
        policy_train_loader = DataLoader(EpisodicDoomDataset(
                episodes=sampled_episodes,
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
            pin_memory=True,
        )
        
        if (epoch > config["global_config"]["warmup"]):
            policy_trainer.fit(agent_model, train_dataloaders=policy_train_loader)
        
            for env, n in zip(config["global_config"]["envs"], config["global_config"]["envs_episodes"]):
                reword, best_path = envs.act_in_env(env, n_episodes=n, frame_skip=config["global_config"]["skip_frame"], max_length=config["global_config"]["max_episode_length"], preprocess=simple_preprocessor_float32, post_process=simple_post_processor)
                print(f"mean reword collected by policy {reword} in env {env}")
                create_video_from_frames(best_path.joinpath("obs"), validation_log.joinpath(f"{env}_{epoch}.mp4"), 15)
                
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()