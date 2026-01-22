import numpy as np
from sumTree import SumTree
from abc import ABC, abstractmethod
from typing import Any

from episode import Transition

class PriorityExperienceReplay(ABC):

    def __init__(self, buffer_size:int):
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
    """
    Adversarial Replay (Curious Replay):
      p = (|L| + eps)^alpha
    Here an 'experience' in your code is an Episode (sequence).
    We prioritize the episode using its loss (recommended: last-step loss).
    """

    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.7,
        eps: float = 1e-2,
        p_max: float = 1e6,
        use_running_min: bool = False,
    ):
        super().__init__(buffer_size)
        # Validate parameters
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        # Initialize parameters
        self.alpha = float(alpha)
        self.eps = float(eps)
        # Initialize PMAX starting from a small value 
        self.PMAX = 1.0
        self.p_max = float(p_max) # upper bound for PMAX
        # trick: use running min to shift losses (usefull to stabilize the loss)
        self.use_running_min = use_running_min
        self.running_min = np.inf

    def store(self, transition) -> int:
        """insert a new entry in the buffer with max priority"""
        idx = self.buffer.buffer_idx
        self.buffer.store(transition, self.PMAX)
        #return the buffer index
        return idx

    def get(self, b: int):
        """Get a batch of b episodes from the buffer"""
        #root of the tree
        P = float(self.buffer.get_priority(0))
        batch = np.ndarray(shape=(b), dtype=np.object_) 
        priority_idx = np.ndarray(shape=(b), dtype=np.int32)
        buffer_idx = np.ndarray(shape=(b), dtype=np.int32)
        #we divide P into b segments (so i'm sure to cover all the range with the random samples)
        s = P / b
        for i in range(b):
            low = s * i
            high = s * (i + 1)
            #sample a value within the segment
            mass = np.random.uniform(low, high)
            #get the corresponding episode
            ep, p_idx, b_idx = self.buffer.get_transition(mass)
            batch[i] = ep
            priority_idx[i] = p_idx
            buffer_idx[i] = b_idx

        return batch, priority_idx, buffer_idx

    def update(self, **params):
        """
        Update priorities of sampled episodes
        Expected:
          - priority_idx: array of SumTree leaf indices (like in CountBased)
          - losses: array shape [b] (episode-loss scalar per sampled episode)
        """
        priority_idx = np.asarray(params["priority_idx"], dtype=np.int32).reshape(-1)
        losses = np.asarray(params["losses"], dtype=np.float32).reshape(-1)

        if len(priority_idx) != len(losses):
            raise ValueError("priority_idx and losses must have same length")
        
        # optionally shift losses by running min
        if self.use_running_min:
            cur_min = float(np.min(losses))
            if cur_min < self.running_min:
                self.running_min = cur_min
            losses = losses - self.running_min
        # compute p = (|L| + eps)^alpha
        new_p = (np.abs(losses) + self.eps) ** self.alpha
        # clip priority
        new_p = np.clip(new_p, 1e-8, self.p_max)

        # update priorities in the sum-tree
        for p_idx, p in zip(priority_idx, new_p):
            self.buffer.update_priority(int(p_idx), float(p))

        # keep PMAX high so new entry episodes are sampled soon
        mx = float(np.max(new_p))
        if mx > self.PMAX:
            self.PMAX = mx

    def info(self):
        """Print debug information: alpha, eps, PMAX, leaf priorities"""
        print(f"[AdversarialReplayPriority] alpha={self.alpha}, eps={self.eps}, PMAX={self.PMAX}")
        print(f"Leaf priorities: {self.buffer.priority.tolist()[self.buffer.size:]}")       


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

    

    
    
    

        



    
    