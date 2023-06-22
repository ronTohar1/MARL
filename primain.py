from primal_master.mapf_gym import MAPFEnv


def main():
    env = MAPFEnv(3)
    x = env._reset(0)
    print(x)
    # env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env._step(action)
        print(obs, reward, done, info)
        # env.render()
    env.close()


if __name__ == '__main__':
    main()