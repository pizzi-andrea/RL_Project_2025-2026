import numpy as np
import torch
import cv2 as cv
import torchvision.transforms.functional as F
from network.controller.Policy import DreamerAgent
from torch import no_grad
from env_info import GamePad
from dda import DynamicDataAllocator
from globals import *

from vizdoom import gymnasium_wrapper
import gymnasium

class EnviromentAdapter:
    def __init__(self, agent:DreamerAgent, memory_manager:DynamicDataAllocator, device:str="cpu"):
        self.agent = agent
        self.device = device
        self.memory_m = memory_manager
        
        self.memory = agent.dreamer_model.rssm.init_state(1, self.device)
        self.gamepad = GamePad()

    def reset_memory_agent(self):
         self.memory = self.agent.dreamer_model.rssm.init_state(1, self.device)

    @no_grad
    def act(self, obs:np.ndarray, user_att:list[float], action, mask_actions=None, local_act_order:list[str]=[]):
        self.gamepad.reset()
        
        frame = F.to_tensor(cv.resize(obs, (160,120))).to(dtype=torch.float32).unsqueeze(0) 
        mask_actions = torch.asarray(mask_actions, dtype=torch.bool).unsqueeze(0) 
        user_att_t = torch.tensor(user_att).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        obs_encode = self.agent.dreamer_model.encode(frame, user_att_t)
        
        self.memory, _ = self.agent.dreamer_model.rssm.observe_step(self.memory, action_t, obs_encode)
        
        latent = torch.cat([self.memory.h, self.memory.z], dim=-1)
        latent = self.agent.dreamer_model.feat_proj(latent)

        _, dist_actions = self.agent.actor(latent, mask_actions)
        
        # Campionamento One-Hot (Differenziabile o stocastico a seconda dell'Actor)
        one_hot_action = dist_actions.sample().squeeze(0) # [Global_Dim]
        
        
        # Otteniamo l'indice globale scelto dall'Actor
        global_action_idx = torch.argmax(one_hot_action, dim=-1).item()

        # GLOBAL_BUTTONS deve essere la lista completa di riferimento del tuo progetto
        GLOBAL_BUTTONS = DoomNames.buttons_names # Esempio
        
        # Identifichiamo il nome del bottone corrispondente all'indice globale
        selected_button_name = GLOBAL_BUTTONS[global_action_idx]
        
        # Cerchiamo la posizione di quel nome nella lista locale dell'ambiente attuale
        try:
            local_action_idx = local_act_order.index(selected_button_name)
        except ValueError:
            # Fallback di sicurezza: se l'actor sceglie un'azione non valida per errore di masking
            print(f"Warning: Actor selected {selected_button_name} not in local_act_order")
            local_action_idx = 0

        return local_action_idx, one_hot_action
    
    @no_grad
    def act_in_env(self, enviroment_id:str, n_episodes:int=1, frame_skip:int=1, max_length:int=500):
        
        print(f"current env {enviroment_id}")
        env = gymnasium.make(enviroment_id, render_mode="rgb_array", max_buttons_pressed=1, frame_skip=frame_skip, max_episode_steps=max_length)
        #action_mapper = ButtonActionMapper.from_env(env)
        
        
        
        local_stat_name    = [v.name for v in env.unwrapped.game.get_available_game_variables()]
        local_buttons_name = [b.name for b in env.unwrapped.game.get_available_buttons()]
        
        # Buffer temporaneo per l'episodio corrente
        current_episode_buffer = []
        episode_rewards = []

        
        
        act_mask = dict.fromkeys(DoomNames.buttons_names, 0)
        for b in local_buttons_name:
            act_mask[b] = 1
        mask_values = [act_mask[name] for name in DoomNames.buttons_names]
        act_mask = np.array(mask_values, dtype=bool)
        
        for episode in range(n_episodes):
            self.reset_memory_agent()
            
            # Reset ambiente
            game, _ = env.reset()
            s = game["screen"]

            game_info = game["gamevariables"]
            
            done = False
            total_reward = 0
            local_action = env.action_space.sample()
            step_count = 0 # Contatore locale per l'episodio
            
            while not done:
                act = DoomMap.local2global(local_action, local_buttons_name, DoomNames.buttons_names)
                att = DoomMap.local2global(game_info, local_stat_name, DoomNames.att_player_names, False)

                
                
                
                # Agisci
                local_action, _ = self.act(s, list(att.values()), action=list(act.values()), mask_actions=act_mask, local_act_order=local_buttons_name)
                
            
                # Esegui step
                game_next, reward, terminated, truncated, info = env.step(local_action)
                s_next = game_next["screen"]
                game_info_next = game_next["gamevariables"]
                done = terminated or truncated

                # Recupera l'azione logica mappata sul gamepad globale per il salvataggio
                #current_action_bool = list(self.gamepad.to_dict().values())

                # Costruzione transizione
                transition = {
                    "obs": cv.resize(s, (160,120)),
                    "ids": step_count,
                    "reward": float(reward),
                    "done": bool(done),
                    
                }

                

                current_episode_buffer.append(transition | act | att)
                s = s_next
                game_info = game_info_next
                total_reward += reward
                step_count += 1
                
                # Aggiornamento stato per il prossimo passo
               
             
            # Fine episodio: salva tramite memory manager
            self.memory_m.alloc_episode(current_episode_buffer, **{"mask": act_mask})
            
            current_episode_buffer.clear() # Pulisci la lista per il prossimo episodio
            episode_rewards.append(total_reward)
            
            print(f"Episode {episode+1}/{n_episodes} finished. Reward: {total_reward}")
        
        env.close()
        return np.mean(episode_rewards).item()