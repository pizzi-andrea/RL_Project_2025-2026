import torch
from torch import nn

##########################################
# taken from the ray framework libraries #
# https://arxiv.org/pdf/1712.05889       #
##########################################

class DreamerV3GRUCell(nn.Module):
    # == Analogous to Danijar's JAX GRU unit code
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = 3 * hidden_size

        # == Linear projection combining input and previous hidden state
        self.linear = nn.Linear(
            input_size + hidden_size,
            self.output_size,
            bias=False,
        )
        # == Using standard initializer as per DreamerV3 specs
        nn.init.xavier_uniform_(self.linear.weight)

        self.layer_norm = nn.RMSNorm(self.output_size, eps=0.001)

    def forward(self, x, h_prev):
        # == Concatenate input with previous state
        x = torch.cat([h_prev, x], dim=-1)
        x = self.linear(x)
        x = self.layer_norm(x)
        
        # == Split into gate components: reset, candidate, update
        reset, cand, update = torch.split(x, self.hidden_size, dim=-1)
        
        # == Apply GRU gates logic
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        
        # == Update hidden state
        h = update * cand + (1 - update) * h_prev
        return h