import torch
from torch import Tensor
from functools import total_ordering

@total_ordering
class Transition:
    """Single transition state - action - user_att"""
    def __init__(self, idx:Tensor, o_t:str, a:Tensor, r:Tensor, idx_n:Tensor, d:Tensor, player_state:Tensor):
        self.idx = idx
        self.idx_n = idx_n
        self.o_t = o_t
        self.action = a
        self.reward = r 
        self.player_state = player_state
        self.done = d
    
    def __eq__(self, value):
        return self.idx == value.idx
    
    def __lt__(self, value):
        return  self.idx < value.idx

class Episode:
    def __init__(self, transition: list["Transition"], max_length: int, short: bool = True):
        if short:
            transition.sort()
        
        transitions = transition[:max_length]
        n = len(transitions)
        
        if n == 0:
            return

        device = torch.device("cpu") 

        self.idx_data = torch.zeros(n, dtype=torch.int16, device=device)
        self.idx_n_data = torch.zeros(n, dtype=torch.int16, device=device)
        self.reward_data = torch.zeros(n, dtype=torch.float32, device=device)
        self.done_data = torch.zeros(n, dtype=torch.bool, device=device)
        
        action_shape = transitions[0].action.shape
        p_state_shape = transitions[0].player_state.shape
        
        self.action_data = torch.zeros((n, *action_shape), dtype=torch.float32, device=device)
        self.player_state_data = torch.zeros((n, *p_state_shape), dtype=torch.float32, device=device)
        
        self.o_t_data = [None] * n

        for i, t in enumerate(transitions):
            self.idx_data[i] = t.idx
            self.idx_n_data[i] = t.idx_n
            self.reward_data[i] = t.reward
            self.done_data[i] = t.done
            self.action_data[i] = t.action
            self.player_state_data[i] = t.player_state
            self.o_t_data[i] = t.o_t

    def __getitem__(self, idx):
        return Transition(
            self.idx_data[idx], 
            self.o_t_data[idx], 
            self.action_data[idx], 
            self.reward_data[idx], 
            self.idx_n_data[idx], 
            self.done_data[idx], 
            self.player_state_data[idx]
        )
    
    def __len__(self):
        return self.idx_data.shape[0]
    


         