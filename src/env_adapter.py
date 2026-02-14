import numpy as np
import torch
import cv2 as cv
import torchvision.transforms.functional as F
from network.controller.Policy import DreamerAgent
from torch import no_grad
from dda import DynamicDataAllocator
from globals import *
from math import inf
from vizdoom import gymnasium_wrapper
import gymnasium

class EnviromentAdapter:
    def __init__(self, agent:DreamerAgent, memory_manager:DynamicDataAllocator, device:str="cpu"):
        self.agent = agent
        self.device = device
        self.memory_m = memory_manager
        
        self.memory = agent.dreamer_model.rssm.init_state(1, self.device)
        #self.gamepad = GamePad()

    def reset_memory_agent(self):
        self.memory = self.agent.dreamer_model.rssm.init_state(1, self.device)
        
    @no_grad()
    def act(self, obs:np.ndarray, user_att:list[float], action, mask_actions=None, local_act_order:list[str]=[]):
        #self.gamepad.reset()
        
        frame = F.to_tensor(cv.resize(obs, (160,120))).to(dtype=torch.float32).unsqueeze(0) 
        mask_actions = torch.asarray(mask_actions, dtype=torch.bool).unsqueeze(0) 
        user_att_t = torch.tensor(user_att).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        obs_encode = self.agent.dreamer_model.encode(frame, user_att_t)
        
        self.memory, _ = self.agent.dreamer_model.rssm.observe_step(self.memory, action_t, obs_encode)

        self.memory.h = self.memory.h.detach()
        self.memory.z = self.memory.z.detach()
        
        latent = torch.cat([self.memory.h, self.memory.z], dim=-1)
        latent = self.agent.dreamer_model.feat_proj(latent)

        _, dist_actions = self.agent.actor(latent, mask_actions)
        
        # Campionamento One-Hot (Differenziabile o stocastico a seconda dell'Actor)
        one_hot_action = dist_actions.sample().squeeze(0) # [Global_Dim]
        #print(mask_actions)
        #print(one_hot_action.shape)
        # Otteniamo l'indice globale scelto dall'Actor
        #print(one_hot_action)
        global_action_idx = torch.argmax(one_hot_action, dim=-1).item()
        #print(global_action_idx)
        #print(mask_actions)
        # GLOBAL_BUTTONS deve essere la lista completa di riferimento del tuo progetto
        GLOBAL_BUTTONS = DoomNames.buttons_names # Esempio
        
        # Identifichiamo il nome del bottone corrispondente all'indice globale
        selected_button_name = GLOBAL_BUTTONS[global_action_idx]
        #print(selected_button_name)
        # Cerchiamo la posizione di quel nome nella lista locale dell'ambiente attuale
        local_action_idx = local_act_order.index(selected_button_name)
        

        return local_action_idx, one_hot_action
    
    @no_grad
    def act_in_env(self, enviroment_id:str, n_episodes:int=1, frame_skip:int=1, max_length:int=500, multibinary:bool=False):
        
        print(f"current env {enviroment_id}")
        env = gymnasium.make(enviroment_id, render_mode="rgb_array", max_buttons_pressed=1, frame_skip=frame_skip, max_episode_steps=max_length)
        #action_mapper = ButtonActionMapper.from_env(env)
        best_episode = -inf
        best_path = ""
        
        
        local_stat_name    = [v.name for v in env.unwrapped.game.get_available_game_variables()]
        
        local_buttons_name = ['NO-OPP'] + [b.name for b in env.unwrapped.game.get_available_buttons()]
        if multibinary:
            local_buttons_name.pop(0)
        
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
            act = OrderedDict()
            step_count = 0 # Contatore locale per l'episodio
            
            while not done:
                if not multibinary:
                    global_idx = DoomMap.reindex_categorical(local_buttons_name[local_action], DoomNames.buttons_names)
                    act_one_hot = DoomMap.one_hot(global_idx, n_classes=len(DoomNames.buttons_names))
                else:
                    act_one_hot = local_action
                

                
                for n, v in zip(DoomNames.buttons_names, act_one_hot):
                    act[n] = v
                
                att = OrderedDict.fromkeys(DoomNames.att_player_names, 0)
                for a, n in zip(game_info, local_stat_name):
                    att[n] = a
 
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
            episode_path = self.memory_m.alloc_episode(current_episode_buffer, **{"mask": act_mask})
            
            current_episode_buffer.clear() # Pulisci la lista per il prossimo episodio
            episode_rewards.append(total_reward)
            
            print(f"Episode {episode+1}/{n_episodes} finished. Reward: {total_reward}")
            if total_reward > best_episode:
                best_path = episode_path


        
        env.close()
        self.reset_memory_agent()
        return np.mean(episode_rewards).item(), best_path