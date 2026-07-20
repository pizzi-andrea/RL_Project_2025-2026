import numpy as np
import torch
import gymnasium
from pathlib import Path

from dd_wrapper import EnviromentWrapperBuilder
from network.controller.Policy import DreamerAgent
from torch import no_grad
from dda import DynamicDataAllocator
from globals import *
from math import inf
from vizdoom import gymnasium_wrapper
from utils import *

class EnviromentAdapter:
    """Wrapper class useful to collect new episods for PER buffers (not use this class to evaluate the model)"""
    def __init__(self, agent:DreamerAgent, memory_manager:DynamicDataAllocator, device:str="cpu"):
        self.agent = agent
        self.device = device
        self.memory_m = memory_manager
    
        self.logger = configure_logger("Environment")
        self.memory = agent.dreamer_model.rssm.init_state(1, self.device)

    def reset_memory_agent(self):
        self.memory = self.agent.dreamer_model.rssm.init_state(1, self.device)
        
    @no_grad()
    def act(self, obs: np.ndarray, user_att: list[float], prev_action: np.ndarray, mask_actions=None, local_act_order: list[str] = [], preprocess=None):

        if preprocess is not None:
            obs_t = preprocess(obs).to(self.device)
        else:
            obs_t = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
            if obs_t.ndim == 3:  # Da [H, W, C] a [C, H, W]
                obs_t = obs_t.permute(2, 0, 1)
            obs_t = obs_t.unsqueeze(0)  # [1, C, H, W]
        
        
        user_att_t = torch.tensor(user_att, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_actions_t = torch.from_numpy(mask_actions).unsqueeze(0).to(self.device) if mask_actions is not None else None
        
        prev_action_t = torch.as_tensor(prev_action, device=self.device, dtype=torch.float32).view(1, -1)
     
        one_hot_action, self.memory = self.agent.act(obs_t, user_att_t, prev_action_t, self.memory, mask_actions_t)
        

        one_hot_action_np = one_hot_action.cpu().numpy() if isinstance(one_hot_action, torch.Tensor) else one_hot_action
        global_action_idx = np.argmax(one_hot_action_np, axis=-1).item()
        
        selected_button_name = DoomNames.buttons_names[global_action_idx]
        local_action_idx = local_act_order.index(selected_button_name)

        return local_action_idx, one_hot_action_np
    
    def _compute_metrics(self, r_list, l_list):
        if not r_list or not l_list:
            return {}
        return {
            "mean_reward": np.mean(r_list).item(),
            "max_reward": np.max(r_list).item(),
            "min_reward": np.min(r_list).item(),
            "std_reward": np.std(r_list).item(),

            "mean_episode_length": np.mean(l_list).item(),
            "max_episode_length": np.max(l_list).item(),
            "min_episode_length": np.min(l_list).item(),
            "std_episode_length": np.std(l_list).item(),
        }
    
    @no_grad()
    def act_in_env(self, enviroment_id: str, n_episodes: int = 1, frame_skip: int = 1, max_length: int = 500, graphics: bool = False, preprocess=None, **args):
        """
        bridge function to run the Dreamer agent in a VizDoom environment and collect metrics.
        """
        self.agent.to(self.device)
        render_mode = "human" if graphics else "rgb_array"
       

        env = EnviromentWrapperBuilder.get_env_from_id(enviroment_id, render_mode, frame_skip, **args)

       
        mapper = DoomMap(env)
        
        episode_rewards = []
        episode_lengths = []
        terminated_episodes = 0
        truncated_episodes = 0
        best_episode = -inf  
        best_path = None    
        
        for episode in range(n_episodes):
            # reset memory of RSSM
            self.reset_memory_agent()
            
            
            # == init episode variables
            game, _ = env.reset()
            s = game["screen"]
            att = mapper.stats_local2global(game["gamevariables"])
            act = mapper.action_local2global(0)
            local_action_hot = [v for v in act.values()]
            ids = 0
            done = False
            r = 0.0
            
            episode = []
            total_reward = 0
            episode_length = 0

            # ==========================
            
            # == main loop to collect a single episode (o_t, a_{t-1}, r_t}) ======================================
            while not done:
                
                # == transition =======================================
                
                transition = {
                    "obs":    s,            # o_{t}
                    "ids":    ids,          # t
                    "reward": r,            # r_{t}
                    "done":   False,        # status
                } | act | att               # a_{t-1} s_{t}
                episode.append(transition)
                
                # == using wrap action act to perform action with Actor model
                local_action, local_action_hot = self.act(
                    s, list(att.values()), prev_action=local_action_hot, 
                    mask_actions=mapper.act_mask, local_act_order=mapper._gym_action_names, preprocess=preprocess
                )
                

                act = mapper.action_local2global(local_action)
                
                
                game_next, r, terminated, truncated, _ = env.step(local_action)
                done = terminated or truncated
                
                s = game_next["screen"]
                att = mapper.stats_local2global(game_next["gamevariables"])
                
                
               
                total_reward += r
                episode_length += 1
                ids += 1

            transition = {
                    "obs":    s,    # s_{T -1}
                    "ids":    ids,  # a_{T-1}
                    "reward": r,    # r_{T-1}      
                    "done":   done,# c_{T-1}
                } | act | att
            episode.append(transition)
    
            # alloc new episode on the disk
            episode_path = self.memory_m.alloc_episode(episode, **{"valid_actions_mask": mapper.act_mask})
            episode.clear()
            
            # accumulate values for metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            if terminated:
                terminated_episodes += 1
            if truncated:
                truncated_episodes += 1

            if total_reward > best_episode:
                best_episode = total_reward
                best_path = episode_path
                
        metrics = self._compute_metrics(episode_rewards, episode_lengths)

        for key, value in metrics.items():
            self.logger.info(f"Aggregate Metric - {key}: {value:.4f}")
        
        if self.agent.logger is not None:
         
            current_step = self.agent.global_step
            self.agent.logger.experiment.add_scalar("episode_return", metrics["mean_reward"], current_step)
            self.agent.logger.experiment.add_scalar("episode_length", metrics["mean_episode_length"], current_step)
            self.agent.logger.experiment.add_scalar("max_reward", metrics["max_reward"], current_step)
        
        self.logger.info(f"Terminated Episodes: {terminated_episodes}/{n_episodes}")
        self.logger.info(f"Truncated Episodes: {truncated_episodes}/{n_episodes}")
        
        env.close()
        self.reset_memory_agent() 
        return np.mean(episode_rewards).item(), best_path