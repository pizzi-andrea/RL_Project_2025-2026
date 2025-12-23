import gymnasium
from vizdoom import gymnasium_wrapper

env = gymnasium.make("VizdoomBasic-v0", render_mode="human")


game = env.unwrapped.game
game.load_config("basic.cfg")  

observation, info = env.reset()


for _ in range(1000):
   
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
