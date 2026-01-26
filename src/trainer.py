from collections import OrderedDict
import os
import tempfile
from PIL import Image
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import no_grad
from env_info import ATT_PLAYER
from priority_functions import CuriosityReplayPriority, CountBasedPriority, AdversarialReplayPriority
from doomData import DoomDataset, EpisodicDoomDataset, collate_episodic_fn
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
import gymnasium
from vizdoom import gymnasium_wrapper



class Trainer(ABC):

    def __init__(self, log_directory):
        self.log_directory = Path(log_directory)
        
    
    @abstractmethod
    def fit(self, path_dataset_train, path_dataset_validation, batch_size, **train_args):
        pass

    def get_model(self):
        return self.model


class VAE_Trainer(Trainer):

    def __init__(self, log_directory, **model_params):
        
        super().__init__(log_directory)
        self.model = VAE(latent_dim=model_params["latent_dim"])
        self.logger = TensorBoardLogger(log_directory, name=False)
    
    def fit(self, path_dataset_train:str, path_dataset_validation:str, batch_size:int, epochs, **train_args):
        data_train = DataLoader(DoomDataset(path_dataset_train, resize=(240,320)), batch_size=batch_size,     shuffle=True, num_workers=os.cpu_count()-1)
        data_valid = DataLoader(DoomDataset(path_dataset_validation, resize=(240,320)), batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()-1)
        trainer = LTrainer(max_epochs=epochs, logger=TensorBoardLogger("VAE_logs"))

        trainer.fit(self.model, train_dataloaders=data_train, val_dataloaders=data_valid)

        return self.model
    
class DreamerXP_Trainer(Trainer):

    def __init__(self, log_directory, **model_params):
        
        super().__init__(log_directory)
        
        self.envs_id = model_params["envs"]
        self.logger = TensorBoardLogger(log_directory, name=False)
        self.dynamic = Dreamer(vae_ckpt_path="VAE_ckpts/epoch=249-step=31500.ckpt")
        

        self.agent = DreamerAgent(self.dynamic.rssm)
        
    @no_grad
    def simulate(self, env_id:str, episods:int, buffer:AdversarialReplayPriority):
        env = gymnasium.make(env_id, max_episode_steps=episods, render_mode="rgb_array")
        stat_name = [v.name for v in env.unwrapped.game.get_available_game_variables()]
        _player = OrderedDict()
        for s in stat_name:
            _player[s] = None
        ids = len(buffer)
        policy = self.controller
        policy.eval()
        for e in range(1, episods+1):
            
            game, _ = env.reset()
            s = game["screen"]
            game_info = game["gamevariables"] 
            done = False
            episode = []
            
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                while not done:
                    a = self.agent.forward(s, game_info)

                    # ---------- step ----------
                    game_next, reward, terminated, truncated, info = env.step(a)
                    done = terminated or truncated

                    s_next = game_next["screen"]
                    game_info_next = game_next["gamevariables"]

                    # ---------- store transition ----------
                    
                    for idx, k in enumerate(_player.keys()):
                        _player[k] = game_info_next[idx]

                    att_diff = ATT_PLAYER.keys() - _player.keys()

                
                    
                    dict_diff = {k: 0 for k in att_diff}
                   
                    
                    ids = torch.tensor(ids)
                    action  = torch.tensor(a)
                    reward = torch.tensor(reward)
                    done = torch.tensor(done, dtype=torch.bool)
                    
                    
                    next_ids = torch.tensor(ids + 1)
                    
                    _user_att = torch.from_numpy([ v for v in (dict_diff | _player).values()])
                    Image.fromarray(s_next).save(tmp_file.name)
                    episode.append(Transition(ids, tmp_file.name, action, reward, next_ids, done, _user_att))
                    ids +=1
            
            buffer.store(Episode(episode))





    def fit(self, path_dataset_train:str, env_id:str, path_dataset_validation:str, batch_size:int, epochs:int, **train_args):

        dataset_train = EpisodicDoomDataset(path_dataset_train, max_length=train_args["episode_length"], priority_buffer=train_args["priority_buffer_train"], preprocessor=train_args["preprocessor"])
        dataset_validation =  EpisodicDoomDataset(path_dataset_validation, max_length=train_args["episode_length"], priority_buffer=train_args["priority_buffer_validation"], preprocessor=train_args["preprocessor"])
        
        buff_train = train_args["priority_buffer_train"]
        buff_validation = train_args["priority_buffer_validation"]
        
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1, collate_fn=collate_episodic_fn)
        loader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1, collate_fn=collate_episodic_fn)
        

        epoch_idx = 0
        train = train_args["train_epochs"]
        validation = train_args["validation_epochs"]
        random_time = train_args["random_time"]
        adapter = PERAdapter(buff_train)
        trainer = LTrainer(callbacks=[adapter], max_epochs=train)

        for iter in range(1, epochs +1):
            

           # -- train phase
            trainer.fit(self.dynamic, loader_train, loader_validation,ckpt_path=self.log_directory)

            for iter_val in range(1,validation+1):
                self.simulate(...)
        
        
        



            

        
        return self.model
        
        

    
    
        