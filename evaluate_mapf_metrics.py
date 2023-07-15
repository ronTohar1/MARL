from simple_maze_env import Maze
from gymnasium.wrappers.flatten_observation import FlattenObservation
from maze_env_wrappers import MazeImageWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C
import gymnasium
import torch as th
import os


def main():
    models_folder = ["models/CnnPolicy/A2C.zip"]
    model = A2C.load(models_folder[0])

    maze_size = 8
    num_agents = 1
    density = 0.2

    env = Maze(maze_size ,num_agents,density=density,max_steps=500)
    env = FlattenObservation(env)

    num_episodes = 100
    metrics = [0,0,0,0]
    for i in range(num_episodes):
        total_reward, sum_of_costs, makespan, is_solved = sum_of_costs_single_episode(model, env, env)
        metrics[0] += total_reward
        metrics[1] += sum_of_costs
        metrics[2] += makespan
        metrics[3] += is_solved
        print(f"episode {i} total_reward {total_reward} sum_of_costs {sum_of_costs} makespan {makespan} is_solved {is_solved}")

    metrics[0] /= num_episodes
    metrics[1] /= num_episodes
    metrics[2] /= num_episodes
    metrics[3] /= num_episodes
    print(f"average total_reward {metrics[0]} sum_of_costs {metrics[1]} makespan {metrics[2]} is_solved {metrics[3]}")


def sum_of_costs_single_episode(agent, env_flatten, env_maze):
    total_reward = 0
    is_solved = False
    sum_of_costs = 0
    makespan = 0
    obs, _ = env_flatten.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        makespan += 1
        sum_of_costs += sum(env_maze.active_agents)
        actions, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_flatten.step(actions)
        total_reward += reward

    is_solved = terminated
    return total_reward, sum_of_costs, makespan, is_solved


        

