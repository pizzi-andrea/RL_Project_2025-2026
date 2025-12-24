import numpy as np
from transition import Transition

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
    
    
                



    def store(self, transition:Transition, priority:float):
        self.buffer[self.buffer_idx] = transition

        self._update(self.buffer_idx)
        self.buffer_idx = (self.buffer_idx + 1)%len(self.buffer)
        

        


        pass

    def clear(self):
        self.priority = np.

    def maximum(self):
        pass

    def minimum(self):
        pass 
    
    def get_priority(self, node:int):
        if node not in self.indexs:
            return 0
        
        return self.priority[node]
    

    
    

    