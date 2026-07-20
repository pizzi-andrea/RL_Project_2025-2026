import gymnasium as gym
import numpy as np
import vizdoom as zd



#TODO: complete for multi-envs
class EnviromentAdapter:
    def __init__(self):
        pass 
    
    @staticmethod
    def get_env_from_id(self, env_id):
        
        match env_id:
            case "VizdoomBasic-v1":
                name = "basic"
            case "VizdoomDeadlyCorridor-v1":
                name = "deadly_corridor"
            case _:
                raise ValueError("cfg not found")
        
        relative =  Path("../configs") / f"{name}.cfg"
        return relative.resolve()

class Wrapper_vz_basic(gym.Wrapper):
    def __init__(self, env, distance_reward_scale=0.05):
        super().__init__(env)
        self.distance_reward_scale = distance_reward_scale
        self.prev_distance = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_distance = self._get_distance_to_target()
        return obs, info

    def step(self, action):
        game = self.env.unwrapped.game
        
        # == Track ammo before action to neutralize the native shoot penalty
        state_before = game.get_state()
        ammo_before = 0
        if state_before is not None:
            ammo_before = game.get_game_variable(zd.GameVariable.AMMO2)
            
        # == Perform step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # == Time penalty to prevent the agent from idling (encourages speed)
        reward -= 0.01
        
        # == Compensate the native -5 shooting penalty to stabilize training
        state_after = game.get_state()
        if state_after is not None:
            ammo_after = game.get_game_variable(zd.GameVariable.AMMO2)
            if ammo_after < ammo_before:
                reward += 4.5  # Neutralizes native penalty down to a mild -0.5
        
        # == Calculate dense distance-based reward delta
        current_distance = self._get_distance_to_target()
        
        if current_distance is not None and self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            shaped_reward = distance_delta * self.distance_reward_scale
            reward += shaped_reward
            self.prev_distance = current_distance
        
        return obs, reward, terminated, truncated, info

    def _get_distance_to_target(self):
        game = self.env.unwrapped.game
        state = game.get_state()
        
        if state is None:
            return None
        
        player_pos = None
        target_pos = None
        
        for obj in state.objects:
            if obj.name == "DoomPlayer":
                player_pos = (obj.position_x, obj.position_y)
            elif obj.name == "RedCard" or "Cacodemon" in obj.name or "Sarge" in obj.name: 
                target_pos = (obj.position_x, obj.position_y)
                
        if player_pos is None:
            player_pos = (game.get_game_variable(zd.GameVariable.POSITION_X), 
                          game.get_game_variable(zd.GameVariable.POSITION_Y))

        if player_pos and target_pos:
            return np.sqrt((player_pos[0] - target_pos[0])**2 + (player_pos[1] - target_pos[1])**2)
        
        return None