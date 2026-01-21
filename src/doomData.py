import os
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype

from episode import Episode, Transition
from priority_functions import PriorityExperienceReplay

class DoomDataset (Dataset):
    """Simple dataset implementation for VAE pre-training"""
    def __init__(self, base_path, resize=(64,64)):
        super().__init__()
        self.paths = []
        self.transformation = Compose([
            Resize(resize),
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True)
        ]) 
        for item in Path(base_path).glob('*'):
            if item.is_dir():
                self.paths.extend([item.joinpath(name) for name in os.listdir(item)])
    

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transformation(image)

    def __len__(self):
        return len(self.paths)


class EpisodicDoomDataset(Dataset):
    """Dataset class for Episodic VizDoom dataset based on PER buffer"""
    def __init__(self, base_path:Path, max_length:int, priority_buffer:PriorityExperienceReplay, preprocessor:Compose=None):
        super().__init__()

        self.buff = priority_buffer
        self.base_path = Path(base_path)
        max_length = max_length
        
        episods = [p for p in self.base_path.iterdir()]
        fix_columns = ["ids","action","reward","done","next_ids"]
        
        if preprocessor:
            self.preprocess = preprocessor
        else:
            self.preprocess = Compose(
                [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True)
                ]
            )
        
        for episode in episods:
        
            obs_path = episode.joinpath("obs")
            t_path = episode.joinpath("transitions.csv")

            episode = []
            t = pd.read_csv(t_path)
            user_att = [att for att in t.columns if att not in fix_columns]
            
            for _, row in t.iterrows():
                ids = torch.tensor(row['ids'])
                action  = torch.tensor(row['action'])
                reward  = torch.tensor(row['reward'])
                done    = torch.tensor(row['done'])
                next_ids = torch.tensor(row['next_ids'])
                
                _user_att = torch.from_numpy(row[user_att].astype(np.float32).values)
                image_path = obs_path.joinpath(f"{ids}.png")
                

                episode.append(Transition(ids, image_path, action, reward, next_ids, done, _user_att))
        
            self.buff.store(Episode(episode, max_length=max_length, short=True))
        


    def __len__(self):
        return len(self.buff)
    
    def __getitem__(self, index):

    
        episode, p_idx, _  = self.buff.get(1)
        t = {
            "obs_idx":[],
            "obs": [],
            "action": [],
            "reward":[],
            "next_obs":[],
            "done": [],
            "user_att":[],
            "priority_idx": p_idx[0]
        }
        
        for transition in episode[0]:
            
            t["obs"].append(self.preprocess(Image.open(transition.o_t).convert("RGB")))
            t["action"].append(transition.action)
            t["done"].append(transition.done)
            t["reward"].append(transition.reward)
            t["user_att"].append(transition.player_state)
            t["obs_idx"].append(transition.idx)
            t["next_obs"].append(transition.idx_n)

        return t
        
            



if __name__ == "__main__":
    dataset_base = "Doom4RSSM/train"
    from priority_functions import CountBasedPriority
    from sumTree import SumTree
    from tqdm.auto import tqdm
    buff = CountBasedPriority(500, 0.1)
    d = EpisodicDoomDataset(dataset_base, 30, buff)
    print(len(d))
    
        


