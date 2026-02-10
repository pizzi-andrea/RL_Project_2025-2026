from abc import abstractmethod
from typing import Any

class AttributeEncoder:
    def __init__(self, att_map:dict, encoder:Any):
        self.att_map = att_map
        self.encoder = encoder

    @abstractmethod
    def encode(self) -> dict:
        
        pass
    
    @abstractmethod
    def decode(self) -> Any:
        pass