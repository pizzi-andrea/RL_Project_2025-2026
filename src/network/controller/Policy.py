import lightning as L
from torch import Tensor, nn
from ..world import Dreamer
class Policy(nn.Module):
    def __init__(self, feat_dim:int, action_space_dim:int):
        super().__init__()
        


        self.policy = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=feat_dim*2),
            nn.LayerNorm(normalized_shape=feat_dim*2),
            nn.Dropout(),
            nn.SiLU(inplace=True),

            nn.Linear(in_features=feat_dim*2, out_features=feat_dim*2),
            nn.LayerNorm(normalized_shape=feat_dim*2),
            nn.Dropout(),
            nn.SiLU(inplace=True),

            nn.Linear(in_features=feat_dim*2, out_features=feat_dim),
            nn.LayerNorm(normalized_shape=feat_dim*2),
            nn.Dropout(),
            nn.SiLU(inplace=True)
        )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=action_space_dim),
            nn.Softmax()
        )

    def forward(self, emb:Tensor):
        action_emb = self.policy(emb)
        action = self.output(action_emb)
        return action
        


class DreamerAgent(L.LightningModule):
    def __init__(self):
        pass