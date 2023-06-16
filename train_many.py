import subprocess
import argparse
from itertools import product


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--maze_size", "-ms", type=int, default=5)
    # parser.add_argument("--num_agents", "-na", type=int, default=2)
    # parser.add_argument("--density", "-d", type=float, default=0, help="density of obstacles")
    # parser.add_argument("--lr", "-lr", type=float, default=0.00001)
    # parser.add_argument("--net_arch", "-na", type=int, nargs="+", default=[64, 64])
    # parser.add_argument("--agent_class", "-ac", choices=['a2c','ppo'], default='a2c')
    # parser.add_argument("num_steps","-ns", type=int, default=100_000)

    parser.add_argument("--policy", "-p", choices=['MlpPolicy', 'CnnPolicy', 'none'], default='none')
    # parser.add_argument("--maze_size", "-ms", type=int, default=5)

    args = parser.parse_args()



    maze_size = [10]
    num_agents = [4]
    density = [0]
    lr = [0.0001,0.001]
    # net_arch = ['[64,64]', '[128,128]', '[128,256]']
    net_arch = ['[64,64]', '[128,128]']
    agent_class = "a2c"
    num_steps = [100_000]
    features_dim = [256]
    # policies = ['MlpPolicy', 'CnnPolicy']
    policies = ['MlpPolicy', 'CnnPolicy']
    if args.policy != 'none':
        policies = [args.policy]
    discount_factors = [0.99]

    counter = 0
    for ms, na, d, lr, net, ns, fd, policy, df in product(maze_size, num_agents, density, lr, net_arch, num_steps, features_dim, policies, discount_factors):
        ac = agent_class

        run_string = f"-ms {ms} -na {na} -d {d} -lr {lr} -net {net} -ac {ac} -ns {ns} -fd {fd} -p {policy} -df {df}"
        r = subprocess.run(['sbatch','train_marl.sh', run_string])
        counter +=1
        print(f"{counter}) {run_string} submitted")

    

if __name__ == "__main__":
    main()