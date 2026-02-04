import gymnasium
from vizdoom import gymnasium_wrapper

env = gymnasium.make("VizdoomBasic-v0")

print("AVAILABLE GAME VARIABLES:")
print([v.name for v in env.unwrapped.game.get_available_game_variables()])

obs_dict, _ = env.reset()
print("\nGAMEVARIABLES AT RESET:")
print(obs_dict["gamevariables"])
print(type(obs_dict["gamevariables"]))
print(len(obs_dict["gamevariables"]))

env.close()
