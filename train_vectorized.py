from stable_baselines3 import A2C, PPO, DDPG, TD3, DQN
from stable_baselines3.common.monitor import Monitor
# from maze_env import Maze
from simple_maze_env import Maze
import numpy as np
from callbacks_log import TensorboardCallback
from gymnasium.wrappers.flatten_observation import FlattenObservation
import gymnasium
import torch as th
import argparse
import random
from maze_env_wrappers import MazeImageWrapper
from maze_cnn import MazeCNN
from stable_baselines3.common.env_checker import check_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_size", "-ms", type=int, default=3)
    parser.add_argument("--num_agents", "-na", type=int, default=1)
    parser.add_argument("--density", "-d", type=float, default=0.5, help="density of obstacles")
    parser.add_argument("--lr", "-lr", type=float, default=0.0001)
    parser.add_argument("--net_arch", "-net", type=str, default='[64, 64]', help="list of hidden layer sizes")
    parser.add_argument("--agent_class", "-ac", choices=['a2c','ppo','dqn'], default='a2c')
    parser.add_argument("--num_steps","-ns", type=int, default=100_000)
    parser.add_argument('--policy','-p',type=str, default='CnnPolicy', choices=['MlpPolicy', 'CnnPolicy'], help='Policy to use for the agent')
    parser.add_argument('--features_dim','-fd',type=int, default=256, help='Number of features for the last layer of the CNN')
    parser.add_argument('--max_steps','-maxs',type=int, default=300, help='Max number of steps in the environment')
    parser.add_argument('--discount_factor','-df',type=float, default=0.99, help='Discount factor for the environment')
    parser.add_argument('--verbose','-v',type=int, default=0, help='Verbosity level for the environment')
    args = parser.parse_args()

    model = MazeCNN
    policy = args.policy
    features_dim = args.features_dim

    net_arch = eval(args.net_arch)
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=net_arch)

    if policy == 'CnnPolicy':
            policy_kwargs["features_extractor_class"] = model
            policy_kwargs["features_extractor_kwargs"] = {"normalized_image":True,
                                                        "features_dim": features_dim,
                                                        "activation_fn": th.nn.ReLU} # All normalized (0 or 1 anyways)

    maze_size = args.maze_size
    num_agents = args.num_agents
    density = args.density
    lr = args.lr
    agent_class = A2C if args.agent_class == 'a2c' else PPO if args.agent_class == 'ppo' else DQN
    num_steps = args.num_steps
    max_steps = args.max_steps
    discount_factor = args.discount_factor
    verbose = args.verbose
    
    env = Maze(maze_size, num_agents ,max_steps=max_steps, density=density)
    env = Monitor(env)
    if policy == 'CnnPolicy':
        env = MazeImageWrapper(env)
    else:         
        env = FlattenObservation(env)


    name = agent_class.__name__ + f"_policy-{policy}"+ str(f"_fd{features_dim}" if policy=="CnnPolicy" else "") + "_lr" + str(lr) + "_net" + str(net_arch) + f"_maze{maze_size}x{maze_size}_agents{num_agents}_density({density})_gamma{discount_factor}"
    agent = agent_class(policy,env , verbose=verbose, tensorboard_log=f"./tensorboard_vec_{maze_size}/", learning_rate=lr, policy_kwargs=policy_kwargs, gamma=discount_factor)
    # agent.learn(total_timesteps=num_steps, log_interval=100, tb_log_name=name )
    agent.learn(total_timesteps=num_steps, log_interval=100, tb_log_name=name)
    agent.save(f"./models/{policy}/{agent_class.__name__}")
    


if __name__ == "__main__":
    main()