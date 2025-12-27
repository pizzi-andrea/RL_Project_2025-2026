from sumTree import SumTree

from abc import ABC, abstractmethod

class PriorityExperienceReplay(ABC):

    def __init__(self, buffer:SumTree):
        super().__init__()
        self.replay = None
        pass
    
    @abstractmethod
    def store(self, transition):
        pass
    
    @abstractmethod
    def _update(self, idx):
        pass 

    @abstractmethod
    def get_data(self):
        """Get next value from buffer"""
        pass


class CountBasedPriority(PriorityExperienceReplay):
    pass 



    
    