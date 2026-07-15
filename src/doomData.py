import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import torch
from pathlib import Path
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from priority_functions import PriorityExperienceReplay

class EpisodicDoomDataset(Dataset):
    def __init__(self, 
                 episodes:list[str], 
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
        self.episodes = episodes
        self.chunk_size = chunk_size # Lunghezza della finestra per il training
        self.action_name = [*action_name]
        self.att_name = [*att_name]
        self.per_weights = per_weights
        self.h = obs_size[0]
        self.w = obs_size[1]
        

    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, index):
        episode_path_m = Path(self.episodes[index]).joinpath(self.csv_name)
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
        is_first_val = (metadata["ids"].iloc[0] == 0)
        
        
        obs_path = metadata["path_obs"]
        
        rewards = metadata["reward"].to_numpy(dtype=np.float16)
        dones = metadata["done"].to_numpy(dtype=bool)

        obs = np.empty( (len(metadata), self.h, self.w, 3), dtype=np.uint8)
     
        priority = self.per_indexs[index]
        with np.load(episode_path_m.parent.joinpath("mask_actions.npz")) as data:
            mask_action = data["mask"]
        
        obs_dir = episode_path_m.parent / "obs"
        last_ok = None
        for idx, p in enumerate(obs_path):
            p = Path(p)
            if not p.is_absolute():
                p = obs_dir / p.name 
            if not p.exists():
                # fallback: ripeti ultimo frame valido (episodio corto / frame mancante)
                if last_ok is None:
                    raise FileNotFoundError(f"Missing first frame: {p}")
                p = last_ok

            img = Image.open(p).convert("RGB")
            obs[idx] = np.array(img, dtype=np.uint8)
            last_ok = p
                    
        
        # pack single observation
        sampled_episode = {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
            "user_att": att_player,
            "priority_idx": priority,
            "mask_action": mask_action,
            "is_first": is_first_val
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
    batch_action = [torch.tensor(b["action"]) for b in batch]
    batch_reward = [torch.tensor(b["reward"]) for b in batch]
    batch_done = [torch.tensor(b["done"]) for b in batch]
    batch_user_att = [torch.tensor(b["user_att"]) for b in batch]
    batch_idx = torch.stack( [torch.tensor(b["priority_idx"],dtype=torch.uint32) for b in batch])
    batch_mask = torch.stack( [torch.as_tensor(b["mask_action"],dtype=torch.bool) for b in batch])
    batch_weight = torch.stack([torch.as_tensor(b["weight"],dtype=torch.float32) for b in batch])
    batch_is_first = torch.stack([torch.as_tensor(b["is_first"],dtype=torch.bool) for b in batch])
    
 
    obs = pad_sequence(batch_obs, batch_first=True)
    action = pad_sequence(batch_action, batch_first=True).float()
    reward = pad_sequence(batch_reward, batch_first=True).float()
    done = pad_sequence(batch_done, batch_first=True, padding_value=1).float()
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
        "continue": 1.0 - done,
        "mask": mask,
        "user_att": user_att,
        "priority_idx": batch_idx,
        "weight": batch_weight,
        "mask_action": batch_mask,
        "is_first": batch_is_first
    }

def collect_episodes_from_buffer(buffer:PriorityExperienceReplay, max_dim:int=5000):
    buff_path, pririty_idx, buffer_idx, weights  = buffer.get(max_dim)
    return buff_path, pririty_idx, buffer_idx, weights 
