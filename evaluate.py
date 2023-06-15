from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from maze_env import Maze
import numpy as np
from callbacks_log import TensorboardCallback
from gymnasium.wrappers.flatten_observation import FlattenObservation
import gymnasium



env = Maze(3,1,max_steps=None, density=0)
env = Monitor(env)
env = FlattenObservation(env)

model = DQN.load("atc_maze")

obs, info = env.reset()
done = False
env.render()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    print(env.action_names(action))
    obs, reward, terminated, truncated , info = env.step(action)
    done = terminated or truncated
    env.render()
    print(obs)
    print("reward:", reward)
    print()