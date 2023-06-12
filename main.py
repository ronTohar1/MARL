from maze_env import Maze

def main():
    env = Maze(4,2,1)
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        print("action = ", action)
        obs, reward, terminated, truncated , info = env.step(action)
        done = terminated or truncated
        print()
    env.close()

if __name__ == "__main__":
    main()