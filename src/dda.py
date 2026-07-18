import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import deque
from shutil import rmtree
from tofolder import dict2folder

class DynamicDataAllocator:
    """
    Advanced class used to manage dynamic dataset for PER buffer.
    It handles disk storage and memory allocation strategies.
    """

    def __init__(self, log_dir: str, per_buffer, prefix: str = "iter", episode_for_subdir: int = 1000):
        self.log_dir = Path(log_dir)
        self.buffer = per_buffer
        self.episode_for_subdir = episode_for_subdir
        
        self.iter_paths = deque()
        self.episodes_indexs = deque()
        
        self.idx = 0
        self.prefix = prefix
        
        if self.log_dir.exists():
            rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True)

    def alloc_episode(self, episode: list[dict], **exstras:np.ndarray):
        """
        Serialize single episode on the disk

        Args:
            episode(list[dict]): list of steps in the episode, each step contain (s_t, a_t, r_t, c_t)
            exstras(np.ndarray[int]): action mask for multiple key configuration
        """
        
        # Restore from previus bukets or create new one 
        if self.idx > 0 and self.episodes_indexs[-1] < self.episode_for_subdir:
            episode_subdir_idx = self.episodes_indexs[-1]
            base_path = self.iter_paths[-1]
            self.episodes_indexs[-1] += 1
        else:
            episode_subdir_idx = 0
            new_path = self.log_dir.joinpath(f"{self.prefix}_{self.idx}")
            new_path.mkdir(parents=True, exist_ok=True)
            
            self.iter_paths.append(new_path)
            self.episodes_indexs.append(1)
            base_path = new_path
            self.idx += 1

        # create save path
        save_path = base_path.joinpath(f"{episode_subdir_idx}")

        # use auxiliary function to save episode as a folder
        dict2folder(episode, save_path)
        if exstras:
            np.savez_compressed(save_path.joinpath("mask_actions.npz"), **exstras)

        # store only the path in the PER buffer
        self.buffer.store(save_path, True)

        # clean list
        episode.clear()

        return save_path