import numpy as np
from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class DoomNames:
    """Global player attribute names for standard VizDoom enviroment"""
    att_player_names: tuple[str] = ('AMMO0', 'AMMO2','AMMOSELECTED_WEAPON', 'ARMOR', 'HEALTH', 'KILLCOUNT', 'SELECTED_WEAPON_AMMO')
    buttons_names: tuple[str]     =    ('ATTACK', 'CROUCH', 'JUMP', 'MOVE_BACKWARD', 'MOVE_FORWARD', 'MOVE_LEFT', 'MOVE_RIGHT', 'RELOAD', 'SELECT_NEXT_WEAPON', 'SELECT_PREV_WEAPON', 'SPEED', 'STRAFE', 'TURN_LEFT', 'TURN_RIGHT', 'USE', 'ZOOM')

@dataclass(frozen=True)
class DoomMap:
    """Global class indexs for standard VizDoom enviroment"""


    @staticmethod
    def local2global(values:list[int|float]|np.ndarray[int|float], names:list[str], global_order:list[str], one_hot:bool=True) -> dict[str, int]:
        
        d = OrderedDict.fromkeys(global_order, 0)
        v = values
        
        if one_hot:
            v = [0]*len(global_order)
            v[values] = 1

        for i, n in enumerate(names):
            d[n] = v[i]
        
        return d
        
    