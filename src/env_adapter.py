import numpy as np
import torch
import gymnasium
from pathlib import Path

from dd_wrapper import Wrapper_vz_basic
from network.controller.Policy import DreamerAgent
from torch import no_grad
from dda import DynamicDataAllocator
from globals import *
from math import inf
from vizdoom import gymnasium_wrapper
from utils import *

class EnviromentAdapter:
    """Specialized class for Dreamer Agent interaction with VizDoom environment"""
    def __init__(self, agent:DreamerAgent, memory_manager:DynamicDataAllocator, device:str="cpu"):
        self.agent = agent
        self.device = device
        self.memory_m = memory_manager
        # Utilizza il logger configurato in stile ROS 2
        self.logger = configure_logger("Environment")
        self.memory = agent.dreamer_model.rssm.init_state(1, self.device)

    def reset_memory_agent(self):
        self.memory = self.agent.dreamer_model.rssm.init_state(1, self.device)
        
    @no_grad()
    def act(self, obs: np.ndarray, user_att: list[float], prev_action: np.ndarray, mask_actions=None, local_act_order: list[str] = [], preprocess=None):
        # 1. Preprocessing e Batching (deve essere [1, C, H, W])
        if preprocess is not None:
            obs_t = preprocess(obs).to(self.device)
        else:
            obs_t = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
            if obs_t.ndim == 3:  # Da [H, W, C] a [C, H, W]
                obs_t = obs_t.permute(2, 0, 1)
            obs_t = obs_t.unsqueeze(0)  # [1, C, H, W]
        
        # 2. Preparazione Attributi e Maschera [1, Dim]
        user_att_t = torch.tensor(user_att, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_actions_t = torch.from_numpy(mask_actions).unsqueeze(0).to(self.device) if mask_actions is not None else None
        
        # Forza la forma [1, action_dim] in modo robusto.
        prev_action_t = torch.as_tensor(prev_action, device=self.device, dtype=torch.float32).view(1, -1)
     
        one_hot_action, self.memory = self.agent.act(obs_t, user_att_t, prev_action_t, self.memory, mask_actions_t)
        
        # Estrae l'azione di output garantendo la compatibilità NumPy
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
    def act_in_env(self, enviroment_id: str, n_episodes: int = 1, frame_skip: int = 1, max_length: int = 500, graphics: bool = False, preprocess=None, post_process=None):
        self.agent.to(self.device)
        render_mode = "human" if graphics else "rgb_array"
        cfg_path = load_cfg_file(enviroment_id)

        if not cfg_path.exists():
            self.logger.error(f"Configuration file for {enviroment_id} environment not found in {cfg_path}")
            raise FileNotFoundError(str(cfg_path))
        else:
            self.logger.info(f"Configuration file for {enviroment_id} environment found in {cfg_path}")

        env_raw = gymnasium.make(
            enviroment_id,
            render_mode=render_mode,
            frame_skip=frame_skip,
            max_buttons_pressed=1,
            max_episode_steps=max_length,
            scenario_config_file=str(cfg_path)
        )

        env = Wrapper_vz_basic(env_raw, 0.05)
        mapper = DoomMap(env)
        
        episode_rewards = []
        episode_lengths = []
        terminated_episodes = 0
        truncated_episodes = 0
        best_episode = -inf  
        best_path = None    
        
        for episode in range(n_episodes):
            self.reset_memory_agent()
            
            # 1. Inizializzazione dello Stato 0 (Reset)
            # --- L'UNICA STRUTTURA CAUSALE CORRETTA ---
            
            # 1. STEP 0 (Reset)
            game, _ = env.reset()
            s = game["screen"]
            current_obs = post_process(s) if post_process is not None else s
            current_att = mapper.stats_local2global(game["gamevariables"])
            
            # Unica azione dummy necessaria (per lo stato iniziale)
            dummy_act = {k: 0 for k in mapper.action_local2global(0).keys()}
            
            current_episode_buffer = [{
                "obs":    current_obs,  # s_0
                "ids":    0, 
                "reward": 0.0,          # Nessun reward a t=0
                "done":   False,        # Non terminato
            } | dummy_act | current_att]
            
            # Setup per l'agente
            prev_act_t = np.zeros((len(DoomNames.buttons_names)), dtype=np.uint8) 
            total_reward = 0
            episode_length = 0
            done = False
            ids = 1
            
            # 2. STEP 1...T (Ciclo di gioco reale)
            while not done:
                # Scegli l'azione a_t basandoti sul frame s_t
                local_action, one_hot_act_t = self.act(
                    s, list(current_att.values()), prev_action=prev_act_t, 
                    mask_actions=mapper.act_mask, local_act_order=mapper._gym_action_names, preprocess=preprocess
                )
                
                # Questa è a_t
                act_global_dict = mapper.action_local2global(local_action)
                
                # Esegui a_t per ottenere s_{t+1}, r_{t+1}, d_{t+1}
                game_next, reward, terminated, truncated, _ = env.step(local_action)
                done = terminated or truncated
                
                s_next = game_next["screen"]
                obs_to_store = post_process(s_next) if post_process is not None else s_next
                att_next = mapper.stats_local2global(game_next["gamevariables"])
                
                # SALVATAGGIO: Scriviamo s_{t+1} assieme all'azione a_t, r_{t+1} e d_{t+1}
                current_episode_buffer.append({
                    "obs":    obs_to_store,      # s_{t+1}
                    "ids":    ids, 
                    "reward": float(reward),     # r_{t+1}
                    "done":   bool(done),        # d_{t+1}
                } | act_global_dict | att_next)
                
                # Avanzamento dei riferimenti
                total_reward += reward
                episode_length += 1
                s = s_next
                current_att = att_next
                prev_act_t = one_hot_act_t
                ids += 1
            
            # FINE DELL'EPISODIO: Nessun salvataggio extra fuori dal while. Nessun dummy finale.
            episode_path = self.memory_m.alloc_episode(current_episode_buffer, **{"mask": mapper.act_mask})
            current_episode_buffer.clear()
            
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
        
        self.logger.info(f"Terminated Episodes: {terminated_episodes}/{n_episodes}")
        self.logger.info(f"Truncated Episodes: {truncated_episodes}/{n_episodes}")
        
        env.close()
        self.reset_memory_agent() 
        return np.mean(episode_rewards).item(), best_path