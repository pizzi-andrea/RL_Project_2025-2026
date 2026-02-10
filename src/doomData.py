import os
import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import torch
from pathlib import Path
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torchvision.io import decode_image
from torchvision.utils import save_image
from tqdm import tqdm
import cv2 as cv
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
    def __init__(self, 
                 episods:list[str], 
                 per_indexs:list, 
                 obs_size:tuple[int,int], 
                 action_name:tuple[str], 
                 att_name:tuple[str],
                 per_weights:list[float],
                 preprocessor=None,
                 csv_name:str = "episode.csv", 
                 chunk_size:int=50
    ):
        
        super().__init__()
        self.preprocessor = preprocessor
        self.per_indexs = per_indexs
        self.csv_name = csv_name
        self.episods = episods
        self.chunk_size = chunk_size # Lunghezza della finestra per il training
        self.action_name = [*action_name]
        self.att_name = [*att_name]
        self.per_weights = per_weights
        self.h = obs_size[0]
        self.w = obs_size[1]
        

    def __len__(self):
        return len(self.episods)
    
    def __getitem__(self, index):
        episode_path_m = Path(self.episods[index]).joinpath(self.csv_name)
        # get length of episode
        total_frames = len(pd.read_csv(episode_path_m, usecols=[0]))
        
        # define chunck
        if total_frames > self.chunk_size:
            start = np.random.randint(0, total_frames - self.chunk_size)
            # skiprows=0 è l'header, quindi saltiamo da 1 a start
            skip = range(1, start + 1) 
            n_rows = self.chunk_size
        else:
            skip = None
            n_rows = total_frames

        # read chunk metadata
        metadata = pd.read_csv(
            episode_path_m, 
            skiprows=skip, 
            nrows=n_rows
        )
       
        actions = metadata[self.action_name].to_numpy(dtype=np.uint8)
        att_player = metadata[self.att_name].to_numpy(dtype=np.uint8)
        
        
        obs_path = metadata["path_obs"]
        
        rewards = metadata["reward"].to_numpy(dtype=np.float16)
        dones = metadata["done"].to_numpy(dtype=np.bool)

        obs = np.empty( (len(metadata), self.h, self.w, 3), dtype=np.uint8)
     
        priority = self.per_indexs[index]
        with np.load(episode_path_m.parent.joinpath("mask_actions.npz")) as data:
            mask_action = data["mask"]
        

        for idx, p in enumerate(obs_path):
            obs[idx] = Image.open(p)
            
        
        # pack single observation
        sampled_episode = {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "user_att": att_player,
            "priority_idx": priority,
            "mask_action": mask_action
        }

        if self.preprocessor is not None:
            
            sampled_episode["obs"] = self.preprocessor(sampled_episode["obs"])

        if self.per_weights is not None:
            weight = self.per_weights[index]
            sampled_episode["weight"] = np.array(weight, dtype=np.float32)
        else:
            sampled_episode["weight"] = np.array(1, dtype=np.float32)
        return sampled_episode
        
def collate_fn(batch):
 
    batch_obs = [b["obs"] for b in batch]
    batch_action = [torch.from_numpy(b["action"]) for b in batch]
    batch_reward = [torch.from_numpy(b["reward"]) for b in batch]
    batch_done = [torch.from_numpy(b["done"]) for b in batch]
    batch_user_att = [torch.from_numpy(b["user_att"]) for b in batch]
    batch_idx = torch.stack( [torch.as_tensor(b["priority_idx"],dtype=torch.uint32) for b in batch])
    batch_mask = torch.stack( [torch.as_tensor(b["mask_action"],dtype=torch.bool) for b in batch])
    batch_weight = torch.stack([torch.as_tensor(b["weight"],dtype=torch.uint32) for b in batch])
    
 
    obs = pad_sequence(batch_obs, batch_first=True)
    action = pad_sequence(batch_action, batch_first=True).float()
    reward = pad_sequence(batch_reward, batch_first=True).float()
    done = pad_sequence(batch_done, batch_first=True).float()
    user_att = pad_sequence(batch_user_att, batch_first=True).float()

    mask = pad_sequence([torch.ones(len(b), dtype=torch.float32) for b in batch_obs], batch_first=True)
    #print(action.shape)
    #print(reward.shape)
    #print(done.shape)
    #exit()
    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "mask": mask,
        "user_att": user_att,
        "priority_idx": batch_idx,
        "weight": batch_weight,
        "mask_action": batch_mask
    }

def collect_episods_from_buffer(buffer:PriorityExperienceReplay, max_dim:int=5000):
    buff_path, pririty_idx, buffer_idx, weights  = buffer.get(max_dim)
    return buff_path, pririty_idx, buffer_idx, weights 


def serialize_episode(episode: list[dict],
                      save_path: str | Path,
                      resize: tuple[int, int] = (160, 120), 
                      ids_name:str="ids",
                      obs_name:str="obs", 
                      act_name:str="action",
                      r_name:str="reward",
                      done_name:str="done",
                      pl_att:str="att_player") -> Path:
    obs = []
    action = []
    reward = []
    done = []
    user_att = []
    ids = []
    
    for transition in episode:
        transition[obs_name] = cv.resize(transition[obs_name], resize)
        obs.append(torch.from_numpy(transition[obs_name]).to(torch.uint8))
        action.append(torch.from_numpy(transition[act_name]).to(torch.float16))
        reward.append(torch.tensor(transition[r_name], dtype=torch.float16))
        done.append(torch.tensor(transition[done_name], dtype=torch.bool))
        user_att.append(torch.from_numpy(transition[pl_att]).to(torch.float16))
        ids.append(torch.tensor(transition[ids_name], dtype=torch.int32))

    

    buff = {
        "obs": torch.stack(obs).permute((0,3,1,2)),
        "action": torch.stack(action),
        "reward": torch.stack(reward),
        "done": torch.stack(done),
        "user_att": torch.stack(user_att),
        "ids": torch.stack(ids)
    }
    
    torch.save(buff, save_path)
    return {"ids_name":"ids", "obs_name":"obs", "act_name":"action", "r_name":"reward", "done_name":"done", "pl_att":"user_att"}


def deserialize_episode(file_path:str|Path) -> dict[str,torch.Tensor]:
    episode_data = torch.load(file_path, map_location='cpu')
    return episode_data