import numpy as np
from dataclasses import dataclass
from collections import OrderedDict


@dataclass(frozen=True)
class DoomNames:
    att_player_names: tuple[str] = (
        'AMMO0', 'AMMO2', 'AMMOSELECTED_WEAPON', 'ARMOR',
        'HEALTH', 'KILLCOUNT', 'SELECTED_WEAPON_AMMO'
    )
    buttons_names: tuple[str] = (
        'NO-OPP', 'ATTACK', 'CROUCH', 'JUMP', 'MOVE_BACKWARD', 'MOVE_FORWARD',
        'MOVE_LEFT', 'MOVE_RIGHT', 'RELOAD', 'SELECT_NEXT_WEAPON',
        'SELECT_PREV_WEAPON', 'SPEED', 'STRAFE', 'TURN_LEFT', 'TURN_RIGHT',
        'USE', 'ZOOM', 'SELECT_WEAPON1', 'SELECT_WEAPON2', 'SELECT_WEAPON3',
        'SELECT_WEAPON4', 'SELECT_WEAPON5', 'SELECT_WEAPON6', 'SELECT_WEAPON7'
    )


class DoomMap:

    def __init__(self, env):
        self._local_buttons: list[str] = [b.name for b in env.unwrapped.game.get_available_buttons()]
        self._local_stats:   list[str] = [v.name for v in env.unwrapped.game.get_available_game_variables()]

        self._gym_action_names: list[str] = []
        for action_vec in env.unwrapped.button_map:
            pressed = [self._local_buttons[i] for i, v in enumerate(action_vec) if v]
            self._gym_action_names.append(pressed[0] if pressed else 'NO-OPP')

        _mask_dict = dict.fromkeys(DoomNames.buttons_names, False)
        for b in self._local_buttons:
            if b in _mask_dict:
                _mask_dict[b] = True
        self.act_mask: np.ndarray = np.array(list(_mask_dict.values()), dtype=bool)

    # ------------------------------------------------------------------

    @property
    def local_buttons(self) -> list[str]:
        return self._local_buttons

    @property
    def local_stats(self) -> list[str]:
        return self._local_stats

    @property
    def n_actions(self) -> int:
        return len(self._gym_action_names)

    # ------------------------------------------------------------------

    def action_local2global(self, action: int) -> dict[str, int]:
        name = self._gym_action_names[action]
        global_act = dict.fromkeys(DoomNames.buttons_names, 0)
        global_act[name] = 1
        return global_act

    def action_global2local(self, global_act: dict[str, int]) -> tuple[int, str]:
        pressed = next((k for k, v in global_act.items() if v == 1 and k in self._gym_action_names), 'NO-OPP')
        return self._gym_action_names.index(pressed), pressed

    # ------------------------------------------------------------------

    def stats_local2global(self, values: list | np.ndarray) -> OrderedDict[str, float]:
        global_stats = OrderedDict.fromkeys(DoomNames.att_player_names, 0.0)
        for i, name in enumerate(self._local_stats):
            if name in global_stats:
                global_stats[name] = float(values[i])
        return global_stats

    def stats_global2local(self, global_stats: dict[str, float]) -> np.ndarray:
        return np.array([global_stats.get(name, 0.0) for name in self._local_stats], dtype=np.float32)