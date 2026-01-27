import argparse
import pandas
from collections import OrderedDict
import os
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from random_policy import Random_Policy
from env_info import ATT_PLAYER, GamePad, GLOBAL_BUTTONS
import gymnasium

from dataclasses import dataclass
from typing import Dict, List

from vizdoom import gymnasium_wrapper

@dataclass
class ButtonActionMapper:
    """
    Rimappa un vettore di azioni MultiBinary (ordine ENV) dentro il GamePad (ordine GLOBALE).
    - env_buttons: ordine dei bottoni come li vede l'ambiente (cambia tra env)
    - global_buttons: ordine canonico (chiavi del GamePad fisso nel dataset
    """
    env_button_names: List[str]
    global_name_to_index: Dict[str, int]


    @classmethod
    # creates mapper from env
    def from_env_and_gamepad(cls, env) -> "ButtonActionMapper":
        # local order: how the env interprets action[i]
        env_buttons = env.unwrapped.game.get_available_buttons()
        env_button_names = [b.name for b in env_buttons]
        # read global button names
        global_name_to_index = {name: i for i, name in enumerate(GLOBAL_BUTTONS)}


        return cls(env_button_names=env_button_names, global_name_to_index=global_name_to_index)

    #from env action to gamepad
    def apply_to_gamepad(self, action, gamepad):
        """
        action: array/list/np/tensor di 0/1, shape [len(env_buttons)]
        Scrive i valori nel GamePad rispettando l'ordine globale.
        """
        # per sicurezza converto a lista di int
        if hasattr(action, "tolist"):
            a = action.tolist()
        else:
            a = list(action)
        
        assert len(a) == len(self.env_button_names), "Action length != env buttons length"

        # setta solo i bottoni che esistono anche nella mappa globale
        for env_idx, pressed in enumerate(a):
            btn_name = self.env_button_names[env_idx] # name of the button in env (es. TURN_LEFT)
            g_idx = self.global_name_to_index.get(btn_name, None)
            if g_idx is None:
                # bottone non presente nella dichiarazione globale viene ignorato
                continue
            gamepad[g_idx] = int(pressed)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rolling procedure')
    parser.add_argument("-d", "--dataset", required=False, default="Vizdoom", type=str)
    parser.add_argument('-n', '--episods', required=True, default=10, type=int)
    parser.add_argument('-s', '--split', required=True, choices=["train", "test", "validation"])
    parser.add_argument('-e', '--env', required=False, default="VizdoomBasic-v0", type=str)
    
    parser.add_argument('--episodic', required=False, action='store_true')
    parser.add_argument("-r", "--reload", required=False, action='store_true' )
    parser.add_argument("-m", "--metadata", required=False, action='store_true')
    parser.add_argument("--skip_frame", required=False, default=1, type=int)
    parser.add_argument("--sticky" ,required=False, action="store_true")
    
    
    args = parser.parse_args()
    env = gymnasium.make(
    args.env,
    render_mode="rgb_array",
    frame_skip=args.skip_frame,
    max_buttons_pressed=0
    )

    if args.sticky:
        sticky = 4
    else:
        sticky = 0

    gamepad = GamePad()
    mapper = ButtonActionMapper.from_env_and_gamepad(env)
    agent = Random_Policy(env.action_space, sticky)

    #check 
    print("ENV buttons:", mapper.env_button_names)
    print("GLOBAL buttons:", GLOBAL_BUTTONS)

    missing = [b for b in mapper.env_button_names if b not in GLOBAL_BUTTONS]
    extra   = [b for b in GLOBAL_BUTTONS if b not in mapper.env_button_names]

    print("Buttons in env but not in global:", missing)
    print("Buttons in global but not in env:", extra)
        


    base_path = Path(args.dataset).joinpath(args.split)
    ids = 0
    init_episode = 0
    if args.reload and base_path.exists():
        if args.episodic:
            init_episode = len(list(base_path.iterdir()))
            #args.episods += init_episode 
        else:    
            ids = sum(1 for _ in base_path.rglob("obs/*.png"))
    else:
        os.makedirs(base_path)
        

    
    buffer = []
    stat_name = [v.name for v in env.unwrapped.game.get_available_game_variables()]
    
    _player = OrderedDict()
    for s in stat_name:
        _player[s] = None
    
    for episode in tqdm(range(init_episode, init_episode + args.episods)):
        
        
        gamepad.reset()
        game, _ = env.reset()
        s = game["screen"]
        game_info = game["gamevariables"] 
        done = False
        reward = 0
        
        if args.episodic:
            work_path = base_path.joinpath(f"{episode}")
            os.makedirs(work_path.joinpath("obs"))
        else:
            os.makedirs(base_path.joinpath("obs"), exist_ok=True)
            work_path = base_path
            
        while not done:
            # ---------- current state ----------
            obs_path = work_path / "obs" / f"{ids}.png"
            Image.fromarray(s).save(obs_path)

            for idx, k in enumerate(_player.keys()):
                _player[k] = game_info[idx]

            # ---------- action ----------
            action = agent.act(s)

            
            # ---------- remap action ----------
            gamepad.reset() #doing so: buttons that are not in the env remain 0
            mapper.apply_to_gamepad(action, gamepad)



            # ---------- step ----------
            game_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            s_next = game_next["screen"]
            game_info_next = game_next["gamevariables"]

            # ---------- store transition ----------
            transition = {
                "ids": ids,
                "reward": reward,
                "done": done,
                "next_ids": ids + 1,
            }

            att_diff = ATT_PLAYER.keys() - _player.keys()

            dict_diff = {k: 0 for k in att_diff}

            buffer.append(transition | _player | gamepad.to_dict() | dict_diff)

            # ---------- avanza ----------
            s = s_next
            game_info = game_info_next
            ids += 1
            gamepad.reset()


            
        
        if args.metadata:
            pandas.DataFrame(buffer).to_csv(work_path.joinpath("transitions.csv"), index=False)
            buffer = []
            ids = 0

    env.close()
    
    if args.metadata and not args.episodic:
        pandas.DataFrame(buffer).to_csv(base_path.joinpath("transitions.csv"), index=False)
        

        

            
