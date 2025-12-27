import numpy as np
from transition import Transition
from typing import Callable

class SumTree:
    """SumTree buffer for PER algorithm"""
    def __init__(self, size:int) -> None:
        self.size = size
        
        self.priority = np.zeros(2*size - 1, dtype=np.float16)
        self.buffer = np.ndarray(shape=size, dtype=np.object_)
        self.indexs = np.arange(start=0, stop=len(self.priority), dtype=np.uint32)
        self.buffer_idx = 0
        

    
    def get_dx(self, parent:int) -> int:
        """Get the right leaf of the parent node if it exists, otherwise return None"""
        i = parent*2 + 2
        if i not in self.indexs:
            return -1
        return i 
    
    def get_sx(self, parent:int) -> int:
        """Get the left leaf of the parent node if it exists, otherwise return None"""
        i = parent*2 + 1
        if i not in self.indexs:
            return -1
        return i 
    
    def get_parent(self, leaf:int) -> int:
        """Get parent node"""
        return (leaf - 1)//2
    
    def is_leaf(self, node:int) -> bool:
        return self.get_dx(node) == -1 and self.get_sx(node) == -1
                

    def _buffer_to_idx(self, idx:int):
        if not self.is_leaf(idx): return -1
        N = self.size
        if idx == 0:
            return 0
        
        highest_bit = 1 << (N.bit_length() - 1)
    
        # 2. Calcola il punto di rottura (quanti elementi vanno nel livello profondo)
        k = 2 * (N- highest_bit)
        
        # 3. Mappatura condizionale
        if idx < k:
            # Bottom level
            # Formula semplificata: (2*n - 1) - k + buffer_index
            return (2 * N - 1) - k + idx
        else:
            # Second to last level
            # Formula semplificata: (n - 1) + (buffer_index - k)
            return (N - 1) + (idx - k)
            

    def store(self, transition:Transition, priority:float):
        self.buffer[self.buffer_idx] = transition

        tree_idx = self._buffer_to_idx(self.buffer_idx)
        self.priority[tree_idx] = priority
        self._update(tree_idx)
        self.buffer_idx = (self.buffer_idx + 1)%len(self.buffer)

        return
        

    def _update(self, idx):
        
        while idx >= 0:
            if self.is_leaf(idx):
                idx = self.get_parent(idx)
                continue
            v_l = self.get_sx(idx)
            v_r = self.get_dx(idx)
            self.priority[idx] = v_l + v_r 
            idx = self.get_parent(idx)
        pass

    def clear(self):
        self.buffer.fill(None)
        self.indexs.fill(0)
    
    def get_priority(self, node:int):
        if node not in self.indexs:
            return 0
        
        return self.priority[node]
    

    
    

    