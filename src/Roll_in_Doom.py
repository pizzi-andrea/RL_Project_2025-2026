import argparse
import pandas
from collections import OrderedDict
import os
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from random_policy import Random_Policy

import gymnasium
from vizdoom import gymnasium_wrapper
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rolling procedure')
    parser.add_argument("-d", "--dataset", required=False, default="Vizdoom", type=str)
    parser.add_argument('-n', '--episods', required=True, default=10, type=int)
    parser.add_argument('-s', '--split', required=True, choices=["train", "test", "validation"])
    parser.add_argument('-e', '--env', required=False, default="VizdoomBasic-v0", type=str)
    parser.add_argument('--episodic', required=False, default=False, type=bool)
    
    
    args = parser.parse_args()
    env = gymnasium.make(
    args.env,
    render_mode="rgb_array",
    max_buttons_pressed=1,
    
    )
    
    
    agent = Random_Policy(env.action_space, 3)
    base_path = Path(args.dataset).joinpath(args.split)

    os.makedirs(base_path)

    ids = 0
    buffer = []
    stat_name = [v.name for v in env.unwrapped.game.get_available_game_variables()]
    
    _player = OrderedDict()
    for s in stat_name:
        _player[s] = None
    
    _env = {
        "ids": 0,
        "action": 0,
        "reward": 0.,
        "done": True
    }

    for episode in tqdm(range(args.episods)):
        
        
        game, _ = env.reset()
        s = game["screen"]
        game_info = game["gamevariables"] 
        done = False
        if args.episodic:
            work_path = base_path.joinpath(f"{episode}")
            os.makedirs(work_path.joinpath("obs"))
        else:
            os.makedirs(base_path.joinpath("obs"), exist_ok=True)
            work_path = base_path
            
        while not done:
            Image.fromarray(s).save(work_path.joinpath("obs").joinpath(f"{ids}.png"))
            
            _env["ids"] = ids
            
            for idx, k in enumerate(_player.keys()):
                _player[k] = game_info[idx]
            
            action = agent.act(s)
            _env["action"] = action
            
            game, reward, terminated, truncated, info = env.step(action)
            
            _env["reward"] = reward
            _env["done"] = (terminated or truncated)
            
            s = game["screen"]
            game_info = game["gamevariables"] 
            done = terminated or truncated
            
            ids += 1

            buffer.append(_env|_player)
        
        if args.episodic:
            pandas.DataFrame(buffer).to_csv(work_path.joinpath("transitions.csv"), index=False)
            buffer = []
            ids = 0

    env.close()
    
    if not args.episodic:
        pandas.DataFrame(buffer).to_csv(base_path.joinpath("transitions.csv"), index=False)
        

        

            
