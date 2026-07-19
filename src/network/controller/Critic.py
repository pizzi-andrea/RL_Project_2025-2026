from torch import nn
from .utils import *
class Critic(nn.Module):
    """Critic Network use to estimate Value functions encoded in buckets spaces and linear spaces"""
    def __init__(self, dim_state, num_buckets, min_buckets=-20, max_buckets=20, hidden_dim=1024):
        """
        """
        super().__init__()
        
        # == buckets parameters
        self.num_buckets = num_buckets
        self.min_buckets = min_buckets
        self.max_buckets = max_buckets

        # == register buckets as buffer for efficiency
        self.register_buffer("buckets", torch.linspace(min_buckets, max_buckets, num_buckets)) 
        
        self.net = self._build_mlp(dim_state, hidden_dim)
        self.buckets_proj = nn.Linear(hidden_dim, num_buckets, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)
   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Assumendo che tu stia importando dreamerv3_normal_initializer altrove
            dreamerv3_normal_initializer(m.weight) 
            if m.bias is not None:
                nn.init.zeros_(m.bias)
   
    def _build_mlp(self, dim_state:int, hidden_dim:int, eps:float=0.001):
        """
        define depth nn for value prediction
        Args:
            dim_state(int):  dimension of world model state
            hidden_dim(int): dimensione of hidden layer
            eps(float): epsilon parameter for LayerNorm 
        """

        return nn.Sequential(
            nn.Linear(dim_state, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2, eps=eps),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.SiLU(),
        )

    def decode_value(self, x):
        """
        Raw decode function, return estimated Value in buckets space (sylog)
        """
        x = self.net(x)
        pred_values_log = self.buckets_proj(x)
        return pred_values_log

    def predict_value(self, x):
        """
        Predict Value from given state x (linear)
        """
        # 1.logits + probs
        pred_values_log = self.decode_value(x)
        pred_values_prob = self.softmax(pred_values_log)
        
        # 2. expected value in symlog space
        v_symlog = (pred_values_prob * self.buckets).sum(dim=-1)
        
        # 3. linear value 
        v_linear = symexp(v_symlog)
        return v_linear

    def forward(self, x):
        pred_values_log = self.decode_value(x)
        pred_values_prob = self.softmax(pred_values_log)
        v_symlog = (pred_values_prob * self.buckets).sum(dim=-1)
        
        return pred_values_log, v_symlog