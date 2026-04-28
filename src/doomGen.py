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
from globals import *
import cv2 as cv


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

    args = parser.parse_args()

    env = gymnasium.make(
        args.env,
        render_mode="rgb_array",
        frame_skip=args.skip_frame,
        max_buttons_pressed=1,
        max_episode_steps=args.horizon
    )

    sticky = 4 if args.sticky else 0
    agent  = Random_Policy(env.action_space, sticky)
    mapper = DoomMap(env)

    print("Local buttons detected:   ", mapper.local_buttons)
    print("Local attributes detected:", mapper.local_stats)
    print("Action mask detected:     ", [ (action, e.item()) for action, e in zip(DoomNames().buttons_names, mapper.act_mask) ] )

    base_path = Path(args.dataset) / args.split

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

    buffer = []

    for episode in tqdm(range(init_episode, init_episode + args.episodes)):
        game, _   = env.reset()
        s         = game["screen"]
        game_info = game["gamevariables"]

        if args.crop:
            s = s[100:, 54:267]
        s = cv.resize(s, (args.resolution_w, args.resolution_h), interpolation=cv.INTER_NEAREST)

        ids         = 0
        done        = False
        act_counter = 0
        act_old     = env.action_space.sample()

        while not done:
            if act_counter >= sticky:
                action      = env.action_space.sample()
                act_old     = action
                act_counter = 1
            else:
                action       = act_old
                act_counter += 1

            act = mapper.action_local2global(action)
            #print( [ (act_name, v) for act_name, v in zip(DoomNames().buttons_names, act.values())] )
            name, local_act = mapper.action_global2local(act)

            

            assert act[local_act] == 1


            game_next, reward, terminated, truncated, info = env.step(action)
            s_next         = game_next["screen"]
            game_info_next = game_next["gamevariables"]
            done           = terminated or truncated

            print(f"{name}|{local_act}|{reward}")

            att = mapper.stats_local2global(game_info_next)

            if args.crop:
                s_next = s_next[100:, 54:267]
            s_next = cv.resize(s_next, (args.resolution_w, args.resolution_h), interpolation=cv.INTER_NEAREST)

            transition = {
                "obs":    s_next,
                "ids":    ids,
                "reward": float(reward),
                "done":   bool(done),
            }
            buffer.append(transition | act | att)

            s         = s_next
            game_info = game_info_next
            ids      += 1

        print("event done")
        if bucket_capacity <= 0:
            current_bucket  = base_path / str(len(buckets))
            os.mkdir(current_bucket)
            buckets.append(current_bucket)
            idx             = 0
            bucket_capacity = bucket_size

        episode_path = current_bucket / str(idx)
        dict2folder(buffer, episode_path)
        np.savez_compressed(episode_path / "mask_actions.npz", mask=mapper.act_mask)

        idx             += 1
        bucket_capacity -= 1
        buffer.clear()

    env.close()
