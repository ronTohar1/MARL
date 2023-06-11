from maze import Maze

def main():
    env = Maze(4,2,0.2)
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        print("action = ", action)
        obs, reward, terminated, truncated , info = env.step(action)
        done = terminated or truncated
        env.render()
        print()
    env.close()

if __name__ == "__main__":
    main()