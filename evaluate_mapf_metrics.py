from simple_maze_env import Maze
from gymnasium.wrappers.flatten_observation import FlattenObservation
from maze_env_wrappers import MazeImageWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import gymnasium
import os
from matplotlib import pyplot as plt


def main():
    models = ["models/winning_models/PPO_policy-MlpPolicy_lr1e-06_net[256, 256]_maze8x8_agents1_density(0.2)_gamma0.99.zip",
              "models/winning_models/PPO_policy-MlpPolicy_lr1e-06_net[512, 512]_maze8x8_agents2_density(0.2)_gamma0.99",
              "models/winning_models/PPO_policy-MlpPolicy_lr1e-06_net[512, 512]_maze8x8_agents3_density(0.2)_gamma0.99",
              "models/winning_models/PPO_policy-MlpPolicy_lr1e-06_net[512, 512]_maze8x8_agents4_density(0.2)_gamma0.99"]
    
    num_agents_arr = [1,2,3,4]
    max_steps = 100
    for index in range(len(models)):

        print("loading ppo model")
        model = PPO.load(models[index])

        print("PPO model loaded successfully")
        maze_size = 8
        num_agents = num_agents_arr[index]
        density = 0.2

        env = Maze(maze_size ,num_agents,density=density,max_steps=max_steps)
        env = FlattenObservation(env)

        num_episodes = 500
        metrics = [[],[],[],[]]
        
        for i in range(num_episodes):
            total_reward, sum_of_costs, makespan, is_solved = sum_of_costs_single_episode(model, env, env)
            metrics[0].append(total_reward)
            metrics[1].append(sum_of_costs)
            metrics[2].append(makespan)
            metrics[3].append(is_solved)
            print(f"episode {i} total_reward {total_reward} sum_of_costs {sum_of_costs} makespan {makespan} is_solved {is_solved}")

        print(f"average total_reward {sum(metrics[0])/len(metrics[0])}")
        print(f"average sum_of_costs {sum(metrics[1])/len(metrics[1])}")
        print(f"average makespan {sum(metrics[2])/len(metrics[2])}")
        print(f"percentage of solved instances {sum(metrics[3])/len(metrics[3]) * 100}%")

        # plot a histogram of how times each value of sum of costs is achieved
        hist_size = max_steps * num_agents
        plt.hist(metrics[1], bins=range(0, hist_size, hist_size//20))
        plt.title("Sum of Costs Frequency")
        plt.xlabel("sum of costs")
        plt.ylabel("frequency")
        # plt.xticks(range(0, hist_size, 5))
        plt.show()

        # plot a histogram of how times each value of makespan is achieved
        plt.hist(metrics[2], bins=range(0, hist_size, hist_size//20))
        plt.title("Makespan Frequency")
        plt.xlabel("makespan")
        plt.ylabel("frequency")
        # plt.xticks(range(0, hist_size, 5))
        plt.show()

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


        

if __name__ == "__main__":
    main()