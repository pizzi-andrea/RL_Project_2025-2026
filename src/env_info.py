

ATT_PLAYER_NAME = ["HEALTH", "AMMO2","SELECTED_WEAPON","SELECTED_WEAPON_AMMO","KILLCOUNT", "ARMOR"]

ATT_PLAYER = {name_att : 0 for name_att in ATT_PLAYER_NAME}

GLOBAL_BUTTONS = [
    "MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD",
    "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE",
    "SELECT_NEXT_WEAPON", "SELECT_PREV_WEAPON",
    "SPEED", "STRAFE",
    "CROUCH", "JUMP",
    "RELOAD", "ZOOM",
] #actions corresponding to the global buttons with the same order

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

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
    def from_env(cls, env) -> "ButtonActionMapper":
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


    def apply_to_env(self, gamepad):
        """
        Converte GamePad (ordine globale) → azione per env (ordine env).
        Ritorna una lista di 0/1 lunga len(env_buttons).
        """
        env_action = [0] * len(self.env_button_names)

        for env_idx, btn_name in enumerate(self.env_button_names):
            g_idx = self.global_name_to_index.get(btn_name, None)
            if g_idx is None:
                continue
            env_action[env_idx] = int(gamepad[g_idx])

        return np.array(env_action, dtype=np.int8)











class GamePad:

    def __init__(self):
        self.reset()
        self._inv_button = {idx: button for idx,button in enumerate(self.button.keys())}
    
    def __getitem__(self, idx:int) -> int:
        return self.button[self._inv_button[idx]]
    
    def __setitem__(self, idx:int, value:int) -> None:
        self.button[self._inv_button[idx]] = value
    
    def to_dict(self) -> dict[str, int]:
        return self.button.copy()
    
    def reset(self):
        self.button = {
            "b1":0,
            "b2":0,
            "b3":0,
            "b4":0,
            "b5":0,
            "b6":0,
            "b7":0,
            "b8":0,
            "b9":0,
            "b10":0,
            "b11":0,
            "b12":0,
            "b13":0,
            "b14":0,
            "b15":0,
            "b16":0,
        }

    def __str__(self):
        return self.button.__str__()

if __name__ == "__main__":
    pad = GamePad()
    pad[5]= 1
    pad[10] = 1
    pad[12] = 1
    print(pad)
    print(pad[5])
    pad.reset()
    print(pad)


def build_user_att_from_env(env, gamevariables: np.ndarray, device=None) -> torch.Tensor:
    """
    Costruisce un vettore [len(ATT_PLAYER_NAME)] nello stesso ordine di ATT_PLAYER_NAME,
    usando solo le variabili che l'env espone. Le mancanti restano a 0.
    """
    names = [v.name for v in env.unwrapped.game.get_available_game_variables()]
    name_to_idx = {n: i for i, n in enumerate(names)}

    att = np.zeros(len(ATT_PLAYER_NAME), dtype=np.float32)
    for j, att_name in enumerate(ATT_PLAYER_NAME):
        i = name_to_idx.get(att_name, None)
        if i is not None and i < len(gamevariables):
            att[j] = float(gamevariables[i])

    t = torch.from_numpy(att)
    if device is not None:
        t = t.to(device)
    return t
