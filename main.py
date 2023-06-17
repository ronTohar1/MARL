from simple_maze_env import Maze
from gymnasium.wrappers.flatten_observation import FlattenObservation
from maze_env_wrappers import MazeImageWrapper

def main():
    env = Maze(4,2,density=0,max_steps=100)
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

if __name__ == "__main__":
    main()