import argparse
import numpy as np

import os

from tqdm.auto import tqdm
from pathlib import Path
from random_policy import Random_Policy
import gymnasium
from doomData import *
from tofolder import *
from vizdoom import gymnasium_wrapper
from gymnasium.envs.registration import register
from globals import *
from dd_wrapper import *
from dd_wrapper import *
import cv2 as cv
from utils import  *


"""Script use to collect multi-envs dataset from Vizdoom"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rolling procedure')
    parser.add_argument("-d", "--dataset", required=False, default="Vizdoom", type=str)
    parser.add_argument('-n', '--episodes', required=True, default=10, type=int)
    parser.add_argument('-s', '--split', required=True, choices=["train", "test", "validation"])
    parser.add_argument('-e', '--env', required=False, default="VizdoomBasic-v0", type=str)
    parser.add_argument('-t', '--horizon', required=False, default=500, type=int)
    parser.add_argument("--skip_frame", required=False, default=1, type=int)
    parser.add_argument("--sticky", required=False, action="store_true")
    parser.add_argument("--resolution_h", required=False, default=120, type=int)
    parser.add_argument("--resolution_w", required=False, default=160, type=int)
    parser.add_argument("--crop", required=False, action="store_true")
    bucket_size = 500
    buckets = []

    logger = configure_logger("doomGen")

    args = parser.parse_args()

    env = EnviromentWrapperBuilder.get_env_from_id(
        args.env,
        gmode="rgb_array",
        skip_frame=args.skip_frame,
        resolution_h=args.resolution_h,
        resolution_w=args.resolution_w,
        crop=args.crop
    )

    sticky = 4 if args.sticky else 0
    agent  = Random_Policy(env.action_space, sticky)
    mapper = DoomMap(env)

    base_path = Path(args.dataset) / args.split

    # if exist try reload from old recording session otherwise create new registration session
    if not base_path.exists():
        os.makedirs(base_path / "0")
        current_bucket  = base_path / "0"
        init_episode    = 0
        bucket_capacity = bucket_size
        idx             = 0
        buckets.append(current_bucket)
    else:
        buckets         = list(base_path.iterdir())
        current_bucket  = buckets[-1]
        init_episode    = len(list(current_bucket.iterdir()))
        idx             = init_episode
        bucket_capacity = bucket_size - init_episode

        print(f"restart from bukets {current_bucket.joinpath(str(idx))}")
        if bucket_capacity <= 0:
            current_bucket  = base_path / str(len(buckets))
            os.mkdir(current_bucket)
            buckets.append(current_bucket)
            idx             = 0
            bucket_capacity = bucket_size
    # -----------------------------------------------------------------------

    buffer = []
    episode_cc = 0
    episode_done = 0
    # main loop to collect episods
    logger.info("start episodes of enviroments [%s]", args.env)
    logger.info("start from episode %i", init_episode)
    


    for episode in tqdm(range(init_episode, init_episode + args.episodes)):
        
       
        game, _   = env.reset()
        s         = game["screen"]
        game_info = game["gamevariables"]
      
        
        act = mapper.action_local2global(0)
        att = mapper.stats_local2global(game_info)
        done = False
        ids = 0
        reward = 0.0
        act_counter = sticky + 1
        
        logger.info("started episode %i of %i", episode_cc, args.episodes)
        
        # 2. Collect single episode
        while not done:

            # store the current transition (s_t, r_t, a_t, c_t, s_{t+1}, done)
            transition = {
                "obs":    s,  # o_t
                "ids":    ids, # a_{t-1}
                "reward": reward, # r_{t-1}
                "done":   done, # c_{t-1}
            }
            buffer.append(transition | act | att)
            
            # == Logica sticky bit
            if act_counter >= sticky:
                action      = env.action_space.sample()
                act_old     = action
                act_counter = 1
            else:
                action       = act_old
                act_counter += 1

            act = mapper.action_local2global(action)
            
            game_next, reward, terminated, truncated, info = env.step(action)
            s_next         = game_next["screen"]
            game_info_next = game_next["gamevariables"]
            done           = terminated or truncated
            att = mapper.stats_local2global(game_info_next)

            game_info = game_info_next
            ids      += 1
        

        # store final timestap (s_t, r_t, a_t, c_t, done)
        transition = {
                "obs":    s,  #
                "ids":    ids,
                "reward": reward,
                "done":   done,
        }
        buffer.append(transition | act | att)

            
        episode_done += (1 - truncated)
        logger.info("end episode %i with status %i", episode_cc, (1 - truncated))

        
        
       
        # sub-divide files in subfolders to avoid big directory
        if bucket_capacity <= 0:
            current_bucket  = base_path / str(len(buckets))
            os.mkdir(current_bucket)
            buckets.append(current_bucket)
            idx             = 0
            bucket_capacity = bucket_size
        # -------------------------------------------------------
        # serialize date on the disk for successive train phase
        episode_path = current_bucket / str(idx)
         
        dict2folder(buffer, episode_path)
        np.savez_compressed(episode_path / "mask_actions.npz", valid_actions_mask=mapper.act_mask)
        # ---------------------------------------------

        idx             += 1
        bucket_capacity -= 1
        episode_cc += 1
        buffer.clear()
    
    logger.info("generation end with %i episoded ended with %i success and %i failed", episode_cc, episode_done,  episode_cc - episode_done)

    env.close()
