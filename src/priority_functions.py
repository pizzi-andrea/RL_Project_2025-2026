import numpy as np
from sumTree import SumTree
from abc import ABC, abstractmethod
from typing import Any

from transition import Transition

class PriorityExperienceReplay(ABC):

    def __init__(self, buffer_size:SumTree):
        super().__init__()
        self.buffer = SumTree(buffer_size)
        pass
    
    @abstractmethod
    def store(self, transition) -> int:
        pass
    
    @abstractmethod
    def update(self, **params):
        pass 

    @abstractmethod
    def get(self, b:int) -> np.ndarray:
        """Get next value from buffer"""
        pass

    @abstractmethod
    def info(self):
        """General information for debugging pourposes"""
        pass

    def __len__(self):
        return len(self.buffer)


class CountBasedPriority(PriorityExperienceReplay):
    def __init__(self, buffer_size:int, beta):
        super().__init__(buffer_size)
        if beta < 0 or beta > 1:
            raise ValueError("Beta value must in [0,1]")
        self.beta = beta
        self.PMAX = 1
        self.counter = np.zeros(shape=(len(self.buffer)), dtype=np.uint32)
    
    def store(self, transition) -> int:
        idx = self.buffer.buffer_idx
        self.counter[idx] = 0
        pi = self.PMAX
        self.buffer.store(transition, pi)
        return idx
    
    def get(self, b:int) -> tuple[list, list[int], list[int]]:

        P = self.buffer.get_priority(0)
        batch = np.ndarray(shape=(b), dtype=np.object_)
        priority_idx = np.ndarray(shape=(b), dtype=np.int32)
        buffer_idx = np.ndarray(shape=(b), dtype=np.int32)
        s = P/b

        for i in range(b):
            low = s * i
            high = s * (i + 1)

            sum = np.random.uniform(low, high)
            t, p_idx, b_idx = self.buffer.get_transition(sum)
            
            self.counter[b_idx] += 1
            batch[i] = t
            priority_idx[i] = p_idx
            buffer_idx[i] = b_idx

            new_p = 1.0 / np.sqrt(1.0 + self.counter[b_idx])
            self.buffer.update_priority(p_idx, new_p)
        
        return batch, priority_idx, buffer_idx
    
    def info(self):
        print(f"Priority probability: {self.buffer.priority.tolist()[self.buffer.size:]}")
        print(f"Number of visits: {self.counter}")

    def update(self, **params):
        """not used in countBasedPriority"""
        pass

        
class AdversarialReplayPriority(PriorityExperienceReplay):
    def __init__(self, buffer: SumTree):
        pass

    def get(self, b: int):
        pass

    def store(self, transition) -> int:
        pass 

    def update(self, **params):
        pass
    def info(self):
        pass 
        

class CuriosityReplayPriority(PriorityExperienceReplay):
    def __init__(self, buffer: SumTree):
        pass

    def get(self, b: int):
        pass

    def store(self, transition) -> int:
        pass 

    def update(self, **params):
        pass 
    def info(self):
        pass      


if __name__ == "__main__":
    replay = CountBasedPriority(SumTree(10), 0.5)
    
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))

    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    replay.store(Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])))
    
    print(replay.counter)

    replay.get(5)
    replay.get(5)
    replay.get(5)
    replay.info()

    

    
    
    

        



    
    