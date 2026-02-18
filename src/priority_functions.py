import numpy as np
from sumTree import SumTree
from abc import ABC, abstractmethod
from typing import Any
from os import remove

class PriorityExperienceReplay(ABC):

    def __init__(self, buffer_size:int):
        super().__init__()
        self.buffer = SumTree(buffer_size)
        pass
    
    @abstractmethod
    def store(self, transition, replace:bool=False) -> int:
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
    
    def clear(self):
        """Clear sumTree structure"""
        self.buffer.clear()
        pass

    def __len__(self):
        return len(self.buffer)


import numpy as np
from typing import Any

class CountBasedPriority(PriorityExperienceReplay):
    def __init__(self, buffer_size: int, beta: float = 0.4):
        super().__init__(buffer_size)
        # Validate beta parameter for Importance Sampling
        if not (0.0 <= beta <= 1.0):
            raise ValueError("Beta value must be in [0,1]")
        
        self.beta = beta
        self.PMAX = 1.0
        
        # Initialize visit counter and replacement tracker
        self.counter = np.zeros(buffer_size, dtype=np.uint32)
        self.replace_buff = np.zeros(buffer_size, dtype=bool)

    def store(self, transition: Any, replace: bool = False) -> int:
        """Insert a new entry into the buffer with maximum priority."""
        idx = self.buffer.buffer_idx
        
        # Handle replacement logic if the slot is occupied and marked for replacement
        if self.buffer.buffer[idx] is not None and self.replace_buff[idx]:
            # Optional: Add explicit cleanup logic here if required by your buffer
            pass
        
        # Reset visit counter for the new entry
        self.counter[idx] = 0
        self.replace_buff[idx] = replace
        
        # Store transition in the underlying buffer with current PMAX
        self.buffer.store(transition, self.PMAX)
        return idx

    def get(self, b: int) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a batch of transitions, calculates IS weights, 
        and updates priorities based on visit counts.
        """
        total_priority = float(self.buffer.get_priority(0))
        N = len(self.buffer)
        
        batch = [None] * b
        priority_idx = np.empty(b, dtype=np.int32)
        buffer_idx = np.empty(b, dtype=np.int32)
        weights = np.empty(b, dtype=np.float32)
        
        # Proportional stratified sampling
        segment = total_priority / b

        for i in range(b):
            # Sample a priority mass within the current segment
            mass = np.random.uniform(segment * i, segment * (i + 1))
            t, p_idx, b_idx = self.buffer.get_transition(mass)
            
            # Increment visit counter for the sampled experience
            self.counter[b_idx] += 1
            
            batch[i] = t
            priority_idx[i] = p_idx
            buffer_idx[i] = b_idx

            # Calculate Importance Sampling (IS) weight
            # p_i = probability of being sampled
            p_i = self.buffer.get_priority(p_idx) / total_priority
            p_i = max(p_i, 1e-8)  # Avoid division by zero
            weights[i] = (1.0 / (N * p_i)) ** self.beta
            
            # Count-Based Logic: Priority decays as the number of visits increases
            new_p = 1.0 / np.sqrt(1.0 + self.counter[b_idx])
            self.buffer.update_priority(p_idx, new_p)
            
            # Keep PMAX updated for future stores
            self.PMAX = max(self.PMAX, new_p)

        # Normalize weights for numerical stability (max weight = 1)
        weights /= (weights.max() + 1e-8)
        
        return batch, priority_idx, buffer_idx, weights

    def info(self):
        """Prints current priority status and visit counts for debugging."""
        print(f"Priority probability: {self.buffer.priority.tolist()[self.buffer.size:]}")
        print(f"Number of visits: {self.counter}")

    def clear(self):
        """Resets the entire priority mechanism and counters."""
        super().clear()
        self.PMAX = 1.0
        self.counter.fill(0)
        self.replace_buff.fill(False)

    def update(self, **params):
        """
        Not explicitly used here as priorities are updated 
        automatically during the 'get' call based on counts.
        """
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

        self.alpha = alpha
        self.eps = eps
        self. p_max = p_max
        self.use_running_min = use_running_min
        self.replace_buff = np.ndarray(len(self.buffer.buffer), dtype=np.bool)
        self.replace_buff.fill(0)
        self.alpha = float(alpha)
        self.eps = float(eps)
        # Initialize PMAX starting from a small value 
        self.PMAX = 1.0
        self.p_max = float(p_max) # upper bound for PMAX
        # trick: use running min to shift losses (usefull to stabilize the loss)
        self.use_running_min = use_running_min
        self.running_min = np.inf

    def store(self, transition:Any, replace:bool=False) -> int:
        """insert a new entry in the buffer with max priority"""
        idx = self.buffer.buffer_idx
        if self.buffer.buffer[idx] is not None and self.replace_buff[idx]:
            remove(self.buffer.buffer[idx])
        
        self.buffer.store(transition, self.PMAX)
        #return the buffer index
        self.replace_buff[idx] = replace
        return idx


    def get(self, b: int, beta: float = 0.4):
        total_priority = float(self.buffer.get_priority(0))
        N = len(self.buffer)

        batch = [None] * b
        priority_idx = np.empty(b, dtype=np.int32)
        buffer_idx = np.empty(b, dtype=np.int32)
        weights = np.empty(b, dtype=np.float32)

        segment = total_priority / b

        for i in range(b):
            mass = np.random.uniform(segment * i, segment * (i + 1))
            ep, p_idx, b_idx = self.buffer.get_transition(mass)

            batch[i] = ep
            priority_idx[i] = p_idx
            buffer_idx[i] = b_idx

            p_i = self.buffer.get_priority(p_idx) / total_priority
            p_i = max(p_i, 1e-8)

            weights[i] = (1.0 / (N * p_i)) ** beta

        weights /= weights.max()
        return batch, priority_idx, buffer_idx, weights



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
    def clear(self):
        super().clear()
        # Initialize parameters
        self.alpha = float(self.alpha)
        self.eps = float(self.eps)
        # Initialize PMAX starting from a small value 
        self.PMAX = 1.0
        self.p_max = float(self.p_max) # upper bound for PMAX
        # trick: use running min to shift losses (usefull to stabilize the loss)
        self.running_min = np.inf
        self.replace_buff.fill(0)

    def info(self):
        """Print debug information: alpha, eps, PMAX, leaf priorities"""
        print(f"[AdversarialReplayPriority] alpha={self.alpha}, eps={self.eps}, PMAX={self.PMAX}")
        print(f"Leaf priorities: {self.buffer.priority.tolist()[:min(len(self.buffer), 10)]}")       


class CuriosityReplayPriority(PriorityExperienceReplay):
    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.7,
        eps: float = 1e-2,
        beta: float = 0.7,
        p_max: float = 1e5,
        c: float = 1e4,
      
    ):
        super().__init__(buffer_size)
        # Validate parameters
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        # Initialize parameters

        self.alpha = alpha
        self.eps = eps
        self.beta = beta
        self.c = c
        # Initialize PMAX starting from a small value 
        self.PMAX = p_max
        self.base_pmax = p_max

        self.replace_buff = np.ndarray(len(self.buffer.buffer), dtype=np.bool)
        self.count_buffer = np.ndarray((2*buffer_size)-1, dtype=np.uint32)
        self.replace_buff.fill(0)
        self.count_buffer.fill(0)
        
        
        
        # trick: use running min to shift losses (usefull to stabilize the loss)
        

    def store(self, transition:Any, replace:bool=False) -> int:
        """insert a new entry in the buffer with max priority"""
        idx = self.buffer.buffer_idx
        if self.buffer.buffer[idx] is not None and self.replace_buff[idx]:
            remove(self.buffer.buffer[idx])
        
        self.buffer.store(transition, self.PMAX)
        self.count_buffer[idx] = 0
        #return the buffer index
        self.replace_buff[idx] = replace
        return idx


    def get(self, b: int, beta: float = 0.4):
        total_priority = float(self.buffer.get_priority(0))
        N = len(self.buffer)

        batch = [None] * b
        priority_idx = np.empty(b, dtype=np.int32)
        buffer_idx = np.empty(b, dtype=np.int32)
        weights = np.empty(b, dtype=np.float32)

        segment = total_priority / b

        for i in range(b):
            mass = np.random.uniform(segment * i, segment * (i + 1))
            ep, p_idx, b_idx = self.buffer.get_transition(mass)

            batch[i] = ep
            priority_idx[i] = p_idx
            buffer_idx[i] = b_idx

            p_i = self.buffer.get_priority(p_idx) / total_priority
            p_i = max(p_i, 1e-8)

            weights[i] = (1.0 / (N * p_i)) ** beta

        weights /= weights.max()
        return batch, priority_idx, buffer_idx, weights



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
        
       
        # compute p = (|L| + eps)^alpha
        
        new_p = (self.c*(self.count_buffer[priority_idx])** self.beta) + (np.abs(losses) + self.eps) ** self.alpha
        # clip priority
        # TODO: 

        # update priorities in the sum-tree
        for p_idx, p in zip(priority_idx, new_p):
            self.count_buffer[p_idx] += 1
            self.buffer.update_priority(int(p_idx), float(p))

        # keep PMAX high so new entry episodes are sampled soon
        mx = float(np.max(new_p))
        if mx > self.PMAX:
            self.PMAX = mx
    
    
    def clear(self):
        super().clear()
      
        # Initialize PMAX starting from a small value 
        self.PMAX = self.base_pmax
        self.replace_buff.fill(0)
        self.count_buffer.fill(0)

    def info(self):
        """Print debug information: alpha, eps, PMAX, leaf priorities"""
        print(f"[AdversarialReplayPriority] alpha={self.alpha}, eps={self.eps}, PMAX={self.PMAX}")
        print(f"Leaf priorities: {self.buffer.priority.tolist()[:min(len(self.buffer), 10)]}")         

    

    
    
    

        



    
    