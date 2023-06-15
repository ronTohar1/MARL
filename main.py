from maze_env import Maze
from gymnasium.wrappers.flatten_observation import FlattenObservation

def main():
    env = Maze(10,2,density=0,max_steps=1000)
    env = FlattenObservation(env)
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        print()
        action = env.action_space.sample()
        print(env.action_names(action))
        obs, reward, terminated, truncated , info = env.step(action)
        # print("reward:", reward)
        total_reward += reward
        done = terminated or truncated
        print()
    env.close()
    print("total reward:", total_reward)

if __name__ == "__main__":
    main()