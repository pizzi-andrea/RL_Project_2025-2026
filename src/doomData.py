import os
import numpy as np
import torch
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from tqdm import tqdm
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
        self.max_length = max_length
        
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
        
        for episode in tqdm(episods, desc="load dataset"):
        
            obs_path = episode.joinpath("obs")
            t_path = episode.joinpath("transitions.csv")

            episode = []
            t = pd.read_csv(t_path)
            user_att = [att for att in t.columns if att not in fix_columns]
            
            for _, row in t.iterrows():
                ids = torch.tensor(row['ids'])
                action  = torch.from_numpy(row[['b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','b15','b16']].astype(np.float32).values)
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
        episode, p_idx, _ = self.buff.get(1)

        obs = []
        action = []
        reward = []
        done = []
        user_att = []

        for transition in episode[0]:
            obs.append(self.preprocess(Image.open(transition.o_t).convert("RGB")))
            action.append(transition.action.unsqueeze(0))
            reward.append(transition.reward)
            done.append(transition.done)
            user_att.append(transition.player_state)

        action = torch.cat(action, dim=0)

        return {
            "obs": torch.stack(obs),              # [ [T1, C, H, W], [T2, C, H, W] ] /-> [B,T,C,H,W]
            "action": action,        # [T]
            "reward": torch.stack(reward),        # [T]
            "done": torch.stack(done),            # [T]
            "user_att": torch.stack(user_att),    # [T, A]
            "priority_idx": torch.tensor(p_idx),
        }
        

def collate_episodic_fn(batch):
    """Collate function for dataloader pytorch"""
    obs = pad_sequence([b["obs"] for b in batch], batch_first=True)        # [B, max_T, C, H, W]
    action = pad_sequence([b["action"] for b in batch], batch_first=True)  # [B, max_T]
    reward = pad_sequence([b["reward"] for b in batch], batch_first=True)
    done = pad_sequence([b["done"] for b in batch], batch_first=True)
    user_att = pad_sequence([b["user_att"] for b in batch], batch_first=True)

    mask = pad_sequence([torch.ones(len(b["obs"])) for b in batch], batch_first=True)

    priority_idx = torch.cat([b["priority_idx"] for b in batch], dim=0)

    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "user_att": user_att,
        "mask": mask,
        "priority_idx": priority_idx
    }


if __name__ == "__main__":
    dataset_base = "Doom4RSSM/train"
    from priority_functions import CountBasedPriority, AdversarialReplayPriority
    buff = AdversarialReplayPriority(500)
    d = EpisodicDoomDataset(dataset_base, 30, buff)

    loader = DataLoader(d, batch_size=16, collate_fn=collate_episodic_fn)

    idx = 0
    for b in loader:
        
        idx +=1
        buff.info()
        print(b["action"].shape)
        p = b["priority_idx"].numpy()
        buff.update(priority_idx=p, losses=np.random.rand(len(p)))
        if idx > 10:
            break

    
        


