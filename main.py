from simple_maze_env import Maze
from gymnasium.wrappers.flatten_observation import FlattenObservation
from maze_env_wrappers import MazeImageWrapper
from sb3_contrib import RecurrentPPO
import gymnasium
import torch as th

def main():
    env = Maze(5,1,density=0.5,max_steps=100)
    # env = FlattenObservation(env)
    env = MazeImageWrapper(env)
    obs, info = env.reset()
    done = False
    total_reward = 0
    curr_agent = 0
    while not done:
        env.render()
        print()
        action = env.action_space.sample()
        print(Maze.action_names([action]))
        print("agent:", curr_agent)
        obs, reward, terminated, truncated , info = env.step(action)
        print(obs)
        # print(obs)
        # print("reward:", reward)
        total_reward += reward
        curr_agent  = (curr_agent + 1) % 2
        done = terminated or truncated
        print()
    env.close()
    print("total reward:", total_reward)


def main2():
    env = gymnasium.make("CartPole-v1")
    maze_size = 16
    env = Maze(maze_size ,1,density=0.1,max_steps=100)
    env = FlattenObservation(env)


    net_arch = [64,64]
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=net_arch)
    policy = "MlpLstmPolicy"

    model = RecurrentPPO(policy, env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0001, gamma=0.99, tensorboard_log=f"./tensorboard_vec_{maze_size}/")
    model.learn(total_timesteps=10000)


if __name__ == "__main__":
    main2()
    # main()