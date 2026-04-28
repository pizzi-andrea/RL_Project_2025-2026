import numpy as np
import torch
import gymnasium

from network.controller.Policy import DreamerAgent
from torch import no_grad
from dda import DynamicDataAllocator
from globals import *
from math import inf
from vizdoom import gymnasium_wrapper


class EnviromentAdapter:
    """Specialized class for Dreamer Agent interaction with VizDoom enviroment"""
    def __init__(self, agent:DreamerAgent, memory_manager:DynamicDataAllocator, device:str="cpu"):
        self.agent = agent
        self.device = device
        self.memory_m = memory_manager
        
        self.memory = agent.dreamer_model.rssm.init_state(1, self.device)
        #self.gamepad = GamePad()

    def reset_memory_agent(self):
        self.memory = self.agent.dreamer_model.rssm.init_state(1, self.device)
        
    @no_grad()
    def act(self, obs:np.ndarray, user_att:list[float], prev_action:np.ndarray, mask_actions=None, local_act_order:list[str]=[], preprocess=None):
        # 1. Preprocessing e Batching (deve essere [1, C, H, W])
        
        if preprocess is not None:
            obs_t = preprocess(obs).to(self.device)
        
        # 2. Preparazione Attributi e Maschera [1, Dim]
        user_att_t = torch.tensor(user_att, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_actions_t = torch.from_numpy(mask_actions).unsqueeze(0).to(self.device)
        prev_action = torch.as_tensor(prev_action, device=self.device, dtype=torch.float32).unsqueeze(0)
     
       
        one_hot_action, self.memory = self.agent.act(obs_t, user_att_t, prev_action, self.memory, mask_actions_t)
        
        global_action_idx = np.argmax(one_hot_action, axis=-1)
        selected_button_name = DoomNames.buttons_names[global_action_idx]
        local_action_idx = local_act_order.index(selected_button_name)

        return local_action_idx, one_hot_action
    
    @no_grad
    def act_in_env(self, enviroment_id:str, n_episodes:int=1, frame_skip:int=1, max_length:int=500, graphics:bool=False, preprocess=None, post_process=None):
        
        self.agent.to(self.device)
        render_mode = "human" if graphics else "rgb_array"
        env = gymnasium.make(enviroment_id, render_mode=render_mode, max_buttons_pressed=1, frame_skip=frame_skip, max_episode_steps=max_length)
        
        # Inizializzazione della utility DoomMap per coerenza con DoomGen
        mapper = DoomMap(env)
        
        episode_rewards = []
        best_episode = -inf  
        best_path = None    
        
        for episode in range(n_episodes):
            self.reset_memory_agent()
            
            game, _ = env.reset()
            s = game["screen"]         # s_t iniziale
            game_info = game["gamevariables"] # info_t iniziale
            
            # Azione precedente inizializzata a zero (NO-OPP)
            prev_act_t = np.zeros((len(DoomNames.buttons_names)), dtype=np.uint8) 
          
            total_reward = 0
            done = False
            ids = 0
            current_episode_buffer = []

            while not done:
                # 1. Preparazione attributi correnti per l'input dell'agente (Stato t)
                att_t_global = mapper.stats_local2global(game_info)

                # 2. L'agente decide l'azione basandosi su s_t e att_t
                # Usiamo mapper._gym_action_names per garantire che l'indice locale 
                # corrisponda esattamente a quello atteso dal wrapper Gymnasium
                local_action, one_hot_act_t = self.act(
                    s, list(att_t_global.values()), 
                    prev_action=prev_act_t, 
                    mask_actions=mapper.act_mask, 
                    local_act_order=mapper._gym_action_names,
                    preprocess=preprocess
                )
                
                # 3. Esecuzione azione nell'ambiente
                game_next, reward, terminated, truncated, _ = env.step(local_action)
                done = terminated or truncated
                
                s_next = game_next["screen"]
                game_info_next = game_next["gamevariables"]

                # 4. Mappatura Risultati (Stato t+1) coerente con DoomGen
                # Otteniamo il dizionario delle azioni globali e degli attributi post-azione
                act_global_dict = mapper.action_local2global(local_action)
                att_next_global = mapper.stats_local2global(game_info_next)

                # Gestione salvataggio osservazione (con eventuale resize/crop)
                obs_to_store = s_next
                if post_process is not None:
                    obs_to_store = post_process(s_next)

                transition = {
                    "obs":    obs_to_store,
                    "ids":    ids, 
                    "reward": float(reward),
                    "done":   bool(done),
                }
                
                # Merge identico a doomGen: [Dati base] + [Azioni Globali] + [Attributi Globali t+1]
                current_episode_buffer.append(transition | act_global_dict | att_next_global)

                # 5. Avanzamento per il prossimo step
                total_reward += reward
                s = s_next
                game_info = game_info_next
                prev_act_t = one_hot_act_t
                ids += 1
            
            # Salvataggio tramite memory manager
            episode_path = self.memory_m.alloc_episode(current_episode_buffer, **{"mask": mapper.act_mask})
            current_episode_buffer.clear() 
            episode_rewards.append(total_reward)
            
            print(f"Episode {episode+1}/{n_episodes} finished. Reward: {total_reward}")
            if total_reward > best_episode:
                best_episode = total_reward
                best_path = episode_path

        env.close()
        self.reset_memory_agent() 
        return np.mean(episode_rewards).item(), best_path