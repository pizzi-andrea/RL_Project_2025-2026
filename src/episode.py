from PIL import Image
from torch import Tensor
import numpy as np

from functools import total_ordering
from collections import deque

@total_ordering
class Transition:
    def __init__(self, idx:Tensor, o_t:str, a:Tensor, r:Tensor, idx_n:Tensor, d:Tensor, player_state:Tensor):
        self.idx = idx
        self.idx_n = idx_n
        self.o_t = o_t
        self.action = a
        self.reward = r 
        self.player_state = player_state
        self.done = d
    
    def __eq__(self, value):
        return self.idx == value.idx
    
    def __lt__(self, value):
        return  self.idx < value.idx

class Episode:
    def __init__(self, transition:list[Transition], max_length:int, short:bool=True):
        if short:
            transition.sort()
        
        self._data = deque(transition, max_length)
    
    def __getitem__(self, idx):
        return self._data[idx]
    
    def __len__(self):
        return len(self._data)

    def add_transition(self, trasition:Transition) -> None:
        self._data.append(trasition)
    
    def remove_transition(self) -> Transition:
        return self._data.popleft()
    


         