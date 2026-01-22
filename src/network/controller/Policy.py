import lightning as L
from ..world import Dreamer
class Policy(L.LightningModule):
    def __init__(self, dreamer:Dreamer):
        super().__init__()
        self.world_model = dreamer
        pass