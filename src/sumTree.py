import numpy as np
from transition import Transition
from typing import Callable

class SumTree:
    """SumTree buffer for PER algorithm"""
    def __init__(self, size:int) -> None:
        self.size = size
        
        self.priority = np.zeros((2*size)-1, dtype=np.float16)
        self.buffer = np.ndarray(shape=size, dtype=np.object_)
        self.buffer_idx = 0
        
    
    def __len__(self) -> int:
        """size of buffer"""
        return self.size

    def get_dx(self, parent:int) -> int:
        """Get the right leaf of the parent node if it exists, otherwise return None"""
        i = parent*2 + 2
        if i < len(self.priority):
            return i
        return -1
    
    def get_sx(self, parent:int) -> int:
        """Get the left leaf of the parent node if it exists, otherwise return None"""
        i = parent*2 + 1
        if i < len(self.priority):
            return i
        return -1 
    
    def get_parent(self, leaf:int) -> int:
        """Get parent node"""
        return (leaf - 1)//2
    
    def is_leaf(self, node:int) -> bool:
        """Check if node is leaf"""
        return node < len(self.priority) and self.get_dx(node) == -1 and self.get_sx(node) == -1
                

    def _buffer_to_idx(self, idx:int) -> int:
        """map index buffer to search tree"""
        if idx < 0 or idx > len(self.buffer): raise IndexError
        return (self.size - 1) + idx
    
    def _idx_to_buffer(self, idx:int) -> int:
        """Map index tree to buffer index"""
        if idx < 0 or idx > len(self.priority): raise IndexError
        return (idx + 1) - self.size

    def store(self, transition:Transition, priority:float):
        """Insert new entry in the buffer"""
        self.buffer[self.buffer_idx] = transition
        tree_idx = self._buffer_to_idx(self.buffer_idx)
        self.priority[tree_idx] = priority
        self._update(tree_idx)
        self.buffer_idx = (self.buffer_idx + 1)%len(self.buffer)
        return
        

    def _update(self, idx):
        """Update tree"""
        while idx >= 0:
            if self.is_leaf(idx):
                idx = self.get_parent(idx)
                continue
            v_l = self.get_sx(idx)
            v_r = self.get_dx(idx)
            self.priority[idx] = self.priority[v_l] + self.priority[v_r] 
            idx = self.get_parent(idx)

    def clear(self):
        """Clear buffer"""
        self.buffer.fill(None)
        self.priority.fill(0)
    
    def get_priority(self, node:int):
        """Get priority correponding to node"""
        if node < 0 or node > len(self.priority):
            return 0.0
        
        return self.priority[node]

    def update_priority(self, start_leaf:int, new_priority:float) -> None:
        if self.is_leaf(start_leaf):
            self.priority[start_leaf] = new_priority
            self._update(start_leaf)
            return
        raise IndexError
    
    def get_transition(self, sum:float):
        
        sum_max = self.priority[0]
        sum = min(sum, sum_max - 1e-4)
        current_idx = 0

        while not self.is_leaf(current_idx):
            sx = self.get_sx(current_idx)

            if sum <= self.priority[sx]:
                current_idx = sx
            else:
                current_idx = self.get_dx(current_idx)
                sum -= self.priority[sx]
            
            if current_idx > len(self.priority):
                raise IndexError("leaf node not valid")
        

        buff_idx = self._idx_to_buffer(current_idx)
    
        return self.buffer[buff_idx], current_idx, buff_idx
            
    

    
    
if __name__ == "__main__":
    tree = SumTree(9)

    tree.store(
        Transition(state=np.array([0,1]), action=1, reward=1.0, next_state=np.array([1,0])),
        10
    )

    
    tree.store(
        Transition(state=np.array([0,1]), action=1, reward=-1.0, next_state=np.array([1,0])),
        5
    )
    tree.store(
        Transition(state=np.array([0,1]), action=1, reward=-1.0, next_state=np.array([1,0])),
        3
    )


    print(tree.get_priority(-10))
    print(tree.is_leaf(-4))
    print(tree.priority)
