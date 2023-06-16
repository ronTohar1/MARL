from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from maze_env import Maze
import numpy as np
from callbacks_log import TensorboardCallback
from gymnasium.wrappers.flatten_observation import FlattenObservation
import gymnasium
from maze_env_wrappers import MazeImageWrapper



env = Maze(10,4,max_steps=100, density=0)
env = Monitor(env)
env = MazeImageWrapper(env)
# env = FlattenObservation(env)

model = A2C.load("models/CnnPolicy/A2C")

obs, info = env.reset()
done = False
env.render()
total_reward = 0
while not done:
    actions, _states = model.predict(obs, deterministic=True)
    print(actions)
    print(Maze.action_names(actions))
    obs, reward, terminated, truncated , info = env.step(actions)
    total_reward += reward
    done = terminated or truncated
    env.render()
    if reward > 0:
        x = input("press enter to continue")
    # print(obs)
    print("reward:", reward)
    print()

print("total reward:", total_reward)