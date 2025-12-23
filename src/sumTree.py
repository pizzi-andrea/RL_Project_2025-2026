
class BinNode:
    def __init__(self, value, data=None) -> None:
        self.sx = None
        self.dx = None
        self.v = value
        self.data = data


class SumTree:
    def __init__(self) -> None:
        self.root = BinNode(0, None)
    
    def insert_dx(self, node:BinNode):
        pass
    
    def insert_sx(self, node:BinNode):
        pass
    
    

    