import argparse
import numpy as np
import pandas
from collections import OrderedDict
import os
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from random_policy import Random_Policy
import gymnasium
from doomData import *
from tofolder import *
from vizdoom import gymnasium_wrapper
from globals import DoomMap, DoomNames







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rolling procedure')
    parser.add_argument("-d", "--dataset", required=False, default="Vizdoom", type=str)
    parser.add_argument('-n', '--episods', required=True, default=10, type=int)
    parser.add_argument('-s', '--split', required=True, choices=["train", "test", "validation"])
    parser.add_argument('-e', '--env', required=False, default="VizdoomBasic-v0", type=str)
    parser.add_argument('-t', '--horizont', required=False, default=500, type=int)
    parser.add_argument("--skip_frame", required=False, default=1, type=int)
    parser.add_argument("--sticky" ,required=False, action="store_true")
    parser.add_argument("--resolution_h", required=False, default=120, type=int)
    parser.add_argument("--resolution_w", required=False, default=160, type=int)
    parser.add_argument("--multibinary", required=False, action="store_true")
    bucket_size = 500
    buckets = []
    
    args = parser.parse_args()
    
    env = gymnasium.make(
    args.env,
    render_mode="rgb_array",
    frame_skip=args.skip_frame,
    max_buttons_pressed=1,
    max_episode_steps=args.horizont
    )

    if args.sticky:
        sticky = 4
    else:
        sticky = 0

    #gamepad = GamePad()
    #mapper = ButtonActionMapper.from_env(env)
    agent = Random_Policy(env.action_space, sticky)

    #check 
    #print("ENV buttons:", mapper.env_button_names)
    #print("GLOBAL buttons:", GLOBAL_BUTTONS)

    #missing = [b for b in mapper.env_button_names if b not in GLOBAL_BUTTONS]
    #extra   = [b for b in GLOBAL_BUTTONS if b not in mapper.env_button_names]

    #print("Buttons in env but not in global:", missing)
    #print("Buttons in global but not in env:", extra)
        


    base_path = Path(args.dataset).joinpath(args.split)
    if not base_path.exists():
        os.makedirs(base_path.joinpath("0"))
        current_bucket = base_path.joinpath("0")
        init_episode = 0
        buckets.append(current_bucket)
        bucket_capacity = 500
        idx = 0
    else:
        buckets = list(base_path.iterdir())
        current_bucket = buckets[-1]
        init_episode = len(list(current_bucket.iterdir()))
        idx = init_episode
        bucket_capacity = bucket_size - init_episode
        if bucket_capacity <= 0:
            current_bucket = Path(base_path.joinpath(f"{len(buckets)}"))
            os.mkdir(current_bucket)
            buckets.append(current_bucket)
            idx = 0
            bucket_capacity = 500
    
    buffer = []
    local_stat_name    =  [v.name for v in env.unwrapped.game.get_available_game_variables()]
    local_buttons_name = ['NO-OPP'] + [b.name for b in env.unwrapped.game.get_available_buttons()]
    if args.multibinary:
        local_buttons_name.pop(0)
        
    act_mask = dict.fromkeys(DoomNames.buttons_names, 0)
    for b in local_buttons_name:
        act_mask[b] = 1
    
    mask_values = [act_mask[name] for name in DoomNames.buttons_names]
    
    act_mask = np.array(mask_values, dtype=bool)
    
    print("Local button dected: ", local_buttons_name)
    print("Local attribute dected", local_stat_name)
    print("buttons mask dected:", act_mask)
    
    
    for episode in tqdm(range(init_episode, init_episode + args.episods)):
        
        ids = 0
        #gamepad.reset()
        game, _ = env.reset()
        s = game["screen"]
        game_info = game["gamevariables"]
        
        action = env.action_space.sample()
        done = False
        reward = 0
        act = OrderedDict()
        act_counter = 0
        act_old = action
        while not done:
            
            # -- encode buttons in global map 
            
            #assert env.action_space.contains(action) 
            
            if not args.multibinary:
                global_idx = DoomMap.reindex_categorical(local_buttons_name[action], DoomNames.buttons_names)
                act_one_hot = DoomMap.one_hot(global_idx, n_classes=len(DoomNames.buttons_names))
            else:
                act_one_hot = action
            

            
            for n, v in zip(DoomNames.buttons_names, act_one_hot):
                act[n] = v
            
            att = OrderedDict.fromkeys(DoomNames.att_player_names, 0)
            for a, n in zip(game_info, local_stat_name):
                att[n] = a
                
            

            # ---------- action ----------
            if  act_counter > sticky:
                action = env.action_space.sample()
                act_old = action
                act_counter = 0
            else:
                action = act_old
                act_counter += 1
            


            # ---------- step ----------
            game_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            s_next = game_next["screen"]
            game_info_next = game_next["gamevariables"]

            # ---------- store transition ----------
            transition = {
                "obs": cv.resize(s, (args.resolution_w,args.resolution_h)),
                "ids": ids,
                "reward": reward,
                "done": done,
            }

            #buffer.append(transition | {"action": np.array(list(gamepad.to_dict().values())), "att_player":np.array(list(player.values()))})
            buffer.append( transition | act | att )

            # ---------- avanza ----------
            s = s_next
            game_info = game_info_next
            ids += 1
            #gamepad.reset()
                    
        
        if bucket_capacity <= 0:
            current_bucket = Path(base_path.joinpath(f"{len(buckets)}"))
            os.mkdir(current_bucket)
            buckets.append(current_bucket)
            idx = 0
            bucket_capacity = 500
        
        dict2folder(buffer, current_bucket.joinpath(f"{idx}"))
        np.savez_compressed(current_bucket.joinpath(f"{idx}").joinpath("mask_actions.npz"), mask=act_mask)
        idx += 1
        bucket_capacity -=1
        buffer.clear()
        # aggiorna anche sopra semplificando la struttura del dataset
    
    env.close()
    
    
        

        

            
