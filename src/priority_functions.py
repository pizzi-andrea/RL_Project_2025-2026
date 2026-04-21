import numpy as np
from sumTree import SumTree
from abc import ABC, abstractmethod
from typing import Any
from shutil import rmtree

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
            rmtree(self.buffer.buffer[idx])
        
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


import numpy as np
from typing import Any
# Assuming PriorityExperience Replay is the base class provided in your framework

class CuriosityReplayPriority(PriorityExperienceReplay):
    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.8,    # Higher alpha at start for aggressive exploration
        eps: float = 1e-2,
        beta: float = 0.4,     # Lower beta at start to allow bias in exploration
        p_max: float = 10.0,   # Lower initial PMAX to avoid "masking" new data
        c: float = 1.0,        # Scaling factor for novelty
    ):
        super().__init__(buffer_size)
        
        # Validation
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        # Hyperparameters
        self.alpha = alpha
        self.eps = eps
        self.beta = beta
        self.c = c
        self.last_updates = None
        # PMAX management: Initialize with a reasonable value to avoid dominating the tree
        self.PMAX = p_max
        self.base_pmax = p_max

        # Buffers
        self.replace_buff = np.ndarray(len(self.buffer.buffer), dtype=bool)
        # count_buffer tracks how many times an episode has been sampled
        self.count_buffer = np.ndarray((2 * buffer_size) - 1, dtype=np.uint32)
        
        self.replace_buff.fill(False)
        self.count_buffer.fill(0)

    def store(self, transition: Any, replace: bool = False) -> int:
        """Inserts a new entry with the current PMAX priority."""
        idx = self.buffer.buffer_idx
        if self.buffer.buffer[idx] is not None and self.replace_buff[idx]:
            # Cleanup logic if using disk-based storage
            rmtree(self.buffer.buffer[idx])
        
        # New episodes start with PMAX to ensure they are sampled at least once
        self.buffer.store(transition, self.PMAX)
        
        # Reset sample counter for the new entry
        tree_idx = self.buffer._buffer_to_idx(idx)
        self.count_buffer[tree_idx] = 0
        
        self.replace_buff[idx] = replace
        return idx

    def update_annealing(self, progress: float):
        """
        Dynamically adjust alpha and beta based on training progress (0.0 to 1.0).
        - Beta should increase to 1.0 to ensure unbiased convergence.
        - Alpha can slightly decrease to stabilize learning in the late stages.
        """
        # Linear annealing for Beta (Importance Sampling correction)
        self.beta = 0.4 + (progress * (1.0 - 0.4))
        
        # Optional: Slightly reduce alpha to focus on reliable samples later on
        # self.alpha = 0.8 - (progress * 0.2)

    def get(self, b: int, beta: float = None):
        """
        Sample a batch of episodes using Stochastic Universal Sampling.
        Uses internal self.beta if no beta is provided.
        """
        current_beta = beta if beta is not None else self.beta
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

            self.count_buffer[p_idx] += 1
            # Probability of sampling this episode
            p_i = self.buffer.get_priority(p_idx) / total_priority
            p_i = max(p_i, 1e-8)

            # Compute Importance Sampling weight to correct the bias
            weights[i] = (1.0 / (N * p_i)) ** current_beta

        # Normalize weights to keep gradients stable
        weights /= (weights.max() + 1e-8)
        return batch, priority_idx, buffer_idx, weights

    def update(self, **params):
        """
        Update priorities based on Prediction Error (Surprise) and novelty.
        Formula: p = (loss + eps)^alpha * (1 / sqrt(count + 1))
        """
        priority_idx = np.asarray(params["priority_idx"], dtype=np.int32).reshape(-1)
        losses = np.asarray(params["losses"], dtype=np.float32).reshape(-1)

        if len(priority_idx) != len(losses):
            raise ValueError("priority_idx and losses must have same length")
       
        # 1. Surprise component (Model error)
        surprise = (np.abs(losses) + self.eps) ** self.alpha
        
        # 2. Novelty component (Saturation bonus)
        # Prevents the model from getting stuck on the same episodes
        novelty_bonus = 1.0 / np.sqrt(self.count_buffer[priority_idx] + 1)

        # 3. Combine and Clip
        # Clipping is crucial to prevent a single outlier from ruining sampling distribution
        new_p = surprise * novelty_bonus
        new_p = np.clip(new_p, 0.01, 100.0) 

        self.last_updates = priority_idx

        for p_idx, p in zip(priority_idx, new_p):
            # Increment sample counter
            
            # Update the SumTree
            self.buffer.update_priority(int(p_idx), float(p))

        # 4. Stable PMAX update
        # We use a moving maximum or a clipped max to avoid PMAX explosion
        current_max = float(np.max(new_p))
        # Smooth PMAX update to avoid sudden jumps
        self.PMAX = 0.9 * self.PMAX + 0.1 * current_max

       

        return self.last_updates, self.buffer.priority[self.last_updates], self.count_buffer[self.last_updates], 
    
    def clear(self):
        super().clear()
        self.PMAX = self.base_pmax
        self.replace_buff.fill(False)
        self.count_buffer.fill(0)
        self.last_updates = None

    def info(self):
        if self.last_updates is None:
            print("No recent updates...")
            return 
        
        
        tree_idxs = self.last_updates
       

        print(f"[Curiosity Replay]  PMAX: {self.PMAX:.2f}")
        print(f"Tree Indices (last batch): {tree_idxs - (self.buffer.size - 1)}")
        print(f"Priorities (leaf values): {np.round(self.buffer.priority[tree_idxs], 2)}")
        print(f"Sample Counts: {self.count_buffer[tree_idxs]}")

    
    
    

        



    
    