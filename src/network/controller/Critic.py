from torch import nn
from .utils import *
class Critic(nn.Module):
    """Network for Critic rule implemented with support at Robust predictor method"""
    def __init__(self, dim_state, num_buckets, min_buckets=-20, max_buckets=20, hidden_dim=1024):
        super().__init__()
        
        self.num_buckets = num_buckets
        self.min_buckets = min_buckets
        self.max_buckets = max_buckets

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
   
    def _build_mlp(self, dim_state, hidden_dim):
        """Helper per costruire MLP senza bias nei layer nascosti e con LayerNorm"""
        return nn.Sequential(
            nn.Linear(dim_state, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2, eps=0.001),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=0.001),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=0.001),
            nn.SiLU(),
        )

    # == LOGICA SEPARATA COMPRESSIONE / LINEARE =================================

    def decode_value(self, x):
        """
        Produce i logit grezzi nello spazio compresso dei bucket.
        Da utilizzare per il calcolo della critic_loss.
        """
        x = self.net(x)
        pred_values_log = self.buckets_proj(x)
        return pred_values_log

    def predict_value(self, x):
        """
        Produce il valore atteso nello spazio lineare.
        Da utilizzare per il calcolo dei lambda returns e dell'advantage.
        """
        # 1. Ottieni i logit e le probabilità
        pred_values_log = self.decode_value(x)
        pred_values_prob = self.softmax(pred_values_log)
        
        # 2. Calcola il valore atteso nello spazio compresso (symlog space)
        v_symlog = (pred_values_prob * self.buckets).sum(dim=-1)
        
        # 3. Decomprimi il valore nello spazio lineare
        v_linear = symexp(v_symlog)
        return v_linear

    # ============================================================================

    def forward(self, x):
        """
        Mantiene la compatibilità con il codice legacy qualora tu debba
        estrarre temporaneamente entrambi i valori (logits e symlog).
        Consiglio di passare gradualmente all'uso esplicito di decode_value e predict_value.
        """
        pred_values_log = self.decode_value(x)
        pred_values_prob = self.softmax(pred_values_log)
        v_symlog = (pred_values_prob * self.buckets).sum(dim=-1)
        
        return pred_values_log, v_symlog