import numpy as np
from sumTree import SumTree
from abc import ABC, abstractmethod

from transition import Transition

class PriorityExperienceReplay(ABC):

    def __init__(self, buffer:SumTree):
        super().__init__()
        self.buffer = buffer
        pass
    
    @abstractmethod
    def store(self, transition) -> int:
        pass
    
    @abstractmethod
    def _update(self, idx):
        pass 

    @abstractmethod
    def get(self, b:int) -> np.ndarray:
        """Get next value from buffer"""
        pass


class CountBasedPriority(PriorityExperienceReplay):
    def __init__(self, buffer: SumTree, beta):
        super().__init__(buffer)
        if beta < 0 or beta > 1:
            raise ValueError("Beta value must in [0,1]")
        self.beta = beta
        self.PMAX = 1
        self.counter = np.zeros(shape=(1, len(self.buffer)), dtype=np.uint32)
    
    def store(self, transition) -> int:
        idx = self.buffer.buffer_idx
        self.counter[idx] = 0
        pi = self.PMAX
        self.buffer.store(transition, pi)
        return idx
    
    def get(self, b:int):
        pass

        

        

        

        
    

        



    
    