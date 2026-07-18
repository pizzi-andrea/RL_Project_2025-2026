import os
import torch
import pandas as pd
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from torch.nn.functional import one_hot


class Callback_one_hot:
    def __init__(self, num_classes:int, buttons_name:list[str]):
        self.num_classes  = num_classes
        self.buttons_name = buttons_name
    
    @staticmethod
    def callback_one_hot(x, num_classes:int, buttons_name:list[str]):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        encode = one_hot(x.long(), num_classes)
        button_dict = {b: encode[i].item() for i, b in enumerate(buttons_name)}
        return button_dict
    
    def __call__(self, x):
        return self.callback_one_hot(x, self.num_classes, self.buttons_name)
        
class Callback_multi_buttons:
    def __init__(self, button_map, button_name):
        self.button_map = button_map
        self.button_name = button_name

    @staticmethod
    def callback_multi_buttons(x, button_map, button_name):
        button_dict = {n: int(b) for b, n in zip(x, button_map)}
        for n in set(button_name) - set(button_dict.keys()):
            button_dict[n] = 0
        return button_dict
    
    def __call__(self, x):
        return self.callback_multi_buttons(x, self.button_map, self.button_name)

class callback_att_to_dict:
    def __init__(self, att_map, att_name):
        self.att_map = att_map
        self.att_name = att_name

    @staticmethod
    def callback_att_to_dict(x, att_map, att_name):
        att_dict = {n: float(b) for b, n in zip(x, att_map)}
        for n in set(att_name) - set(att_dict.keys()):
            att_dict[n] = 0.0
        return att_dict

    def __call__(self, x):
        return self.callback_att_to_dict(x, self.att_map, self.att_name)

def dict2folder(episode:list[dict], out_dir:str|Path, button_encoder=None, att_player_encoder=None, csv_name:str="episode.csv"):
    
    out_dir = Path(out_dir)
    obs_dir = out_dir.joinpath("obs")
    obs_dir.mkdir(parents=True,exist_ok=True)
    
    for idx, _ in enumerate(episode):
        
        frame_path = obs_dir.joinpath(f"{idx}.png")
        Image.fromarray(episode[idx]["obs"]).save(frame_path)
        episode[idx].pop("obs")
        episode[idx]["path_obs"] = str(frame_path)

    
        if button_encoder:
            episode[idx] = episode[idx] | button_encoder(episode[idx]["action"])
            episode[idx].pop("action")

        if att_player_encoder:
            episode[idx]  = episode[idx] | att_player_encoder(episode[idx]["att_player"])
            episode[idx].pop("att_player")
    df = pd.DataFrame(episode)
    df.to_csv(out_dir.joinpath(csv_name), index=False, sep=",")
    return episode
