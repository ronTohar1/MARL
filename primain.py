from primal_master.mapf_gym import MAPFEnv


def main():
    env = MAPFEnv(3)
    x = env.reset()
    print(x)
    # env.render()
    done = False
    while not done:
        # print(env.action_space)
        action = env.action_space.sample()
        # action = tuple([action[0]+1, action[1]])
        print("Action", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)
        # env.render()
    env.close()


if __name__ == '__main__':
    main()