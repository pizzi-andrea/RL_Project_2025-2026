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
        # env definition #######################################################################################################################
        render_mode = "human" if graphics else "rgb_array"
        env = gymnasium.make(enviroment_id, render_mode=render_mode, max_buttons_pressed=1, frame_skip=frame_skip, max_episode_steps=max_length)
        ########################################################################################################################################
        
        # local mapping ###############################################################################
        local_stat_name = [v.name for v in env.unwrapped.game.get_available_game_variables()]
        local_buttons_name = ['NO-OPP'] + [b.name for b in env.unwrapped.game.get_available_buttons()]
        episode_rewards = []
        ###############################################################################################

        # define masks ########################################################################
        act_mask_dict = dict.fromkeys(DoomNames.buttons_names, 0)
        for b in local_buttons_name: act_mask_dict[b] = 1
        act_mask = np.array([act_mask_dict[n] for n in DoomNames.buttons_names], dtype=bool)
        #######################################################################################

        best_episode = -inf  
        best_path = None    
        
        for episode in range(n_episodes):
            # single episode variables ###########
            self.reset_memory_agent()
            
            # init env
            game, _ = env.reset()
            s = game["screen"]
            game_info = game["gamevariables"]
            prev_act_t = np.zeros((len(DoomNames.buttons_names)), dtype=np.uint8) # NO-OPP
          
            total_reward = 0
            done = False
            ids = 0
            ###

            current_episode_buffer = []
            ####################################
            
            # Inizializziamo l'azione precedente come zero
            
            while not done:
                # encode attributes from env #############################
                att = OrderedDict.fromkeys(DoomNames.att_player_names, 0)
                for a, n in zip(game_info, local_stat_name): att[n] = a
                ##########################################################

                # use agent to performe one-step action ##################
                local_action, one_hot_act_t = self.act(
                    s, list(att.values()), 
                    prev_action=prev_act_t, 
                    mask_actions=act_mask, 
                    local_act_order=local_buttons_name,
                    preprocess=preprocess
                )
                ##########################################################
                
                # convert action for serialization
                act_dict = OrderedDict(zip(DoomNames.buttons_names, one_hot_act_t))

                # collect new step
                game_next, reward, terminated, truncated, _ = env.step(local_action)
                done = terminated or truncated

                if post_process is not None:
                    s = post_process(s)

                # 4. Store (Sincronizzazione doomGen: s_t, a_t, att_t, ids_t)
                transition = {
                    "obs": s,
                    "ids": ids, 
                    "reward": reward,
                    "done": bool(done),
                }
                current_episode_buffer.append(transition | dict(act_dict) | dict(att))

                # 5. Avanzamento
                total_reward += reward
                s = game_next["screen"]
                game_info = game_next["gamevariables"]
                prev_act_t = one_hot_act_t # a_t diventa a_t-1 per il prossimo step
                ids += 1
            
            # Salvataggio tramite memory manager
            episode_path = self.memory_m.alloc_episode(current_episode_buffer, **{"mask": act_mask})
            
            current_episode_buffer.clear() 
            episode_rewards.append(total_reward)
            
            print(f"Episode {self.memory_m.episodes_indexs[-1]} of {episode}/{n_episodes} finished. Reward: {total_reward}")
            
            # Aggiornamento del miglior episodio per il logging/checkpoint
            if total_reward > best_episode:
                best_episode = total_reward
                best_path = episode_path

     
        env.close()
       
        self.reset_memory_agent() 
        return np.mean(episode_rewards).item(), best_path