import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import deque
from shutil import rmtree


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
        self.episods_indexs = deque()
        
        self.idx = 0
        self.prefix = prefix
        
        if self.log_dir.exists():
            rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True)

    def alloc_episode(self, episode: list[dict], **exstras:np.ndarray):
        """
        
        """
        
        if self.idx > 0 and self.episods_indexs[-1] < self.episode_for_subdir:
            episode_subdir_idx = self.episods_indexs[-1]
            base_path = self.iter_paths[-1]
            self.episods_indexs[-1] += 1
        else:
            episode_subdir_idx = 0
            new_path = self.log_dir.joinpath(f"{self.prefix}_{self.idx}")
            new_path.mkdir(parents=True, exist_ok=True)
            
            self.iter_paths.append(new_path)
            self.episods_indexs.append(1)
            base_path = new_path
            self.idx += 1

        
        save_path = base_path.joinpath(f"{episode_subdir_idx}")

        # Serializza e salva su disco usando la logica di doomData
        dict2folder(episode, save_path)
        if exstras:
            np.savez_compressed(save_path.joinpath("mask_actions.npz"), **exstras)

        # Memorizza il path nel buffer PER
        self.buffer.store(save_path, True)

        episode.clear()

        return save_path