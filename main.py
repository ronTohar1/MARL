# import gymnasium as gym
import pogema
from pogema import pogema_v0
from my_config import grid_config

mapconfig = grid_config
# for f in mapconfig:
    # print(f)

env = pogema_v0(grid_config=mapconfig)
# env = gym.make("Pogema-v0")

obs = env.reset()

# done is an array of size num_agents [False, True, ...]
# obs is 
while True:
    # Using random policy to make actions
    obs, reward, done, info = env.step(env.sample_actions())
    env.render()
    # print(done)   

    for obsw in obs:
        print(obsw["obstacles"])
    # if all(terminated) or all(truncated):
    if all(done):
        break