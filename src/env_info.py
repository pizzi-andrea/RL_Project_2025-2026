


ATT_PLAYER = {
    # --- STATI VITALI ---
    "HEALTH": 0.0,
    "AMMO2": 0.0,
    "SELECTED_WEAPON": 0,
    "SELECTED_WEAPON_AMMO": 0,
    "KILLCOUNT":0,
    "ARMOR":0.0
}

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