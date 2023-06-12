from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from maze_env import Maze
import numpy as np
from callbacks_log import TensorboardCallback


env = Maze(10,5)
# env = Monitor(env)
model = A2C("MlpPolicy", env, verbose=2, tensorboard_log="./tensorboard_vec/",)
model.learn(total_timesteps=10000, log_interval=50, tb_log_name="ppo_maze", callback=TensorboardCallback())
# model.save("dqn_cartpole")


# obs, info = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done , info = env.step(action)