import subprocess
import argparse
from itertools import product


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--maze_size", "-ms", type=int, default=5)
    # parser.add_argument("--num_agents", "-na", type=int, default=2)
    # parser.add_argument("--density", "-d", type=float, default=0, help="density of obstacles")
    # parser.add_argument("--lr", "-lr", type=float, default=0.00001)
    # parser.add_argument("--net_arch", "-na", type=int, nargs="+", default=[64, 64])
    # parser.add_argument("--agent_class", "-ac", choices=['a2c','ppo'], default='a2c')
    # parser.add_argument("num_steps","-ns", type=int, default=100_000)
    # args = parser.parse_args()



    maze_size = [16]
    num_agents = [4]
    density = [0.3]
    lr = [0.0001,0.00001]
    net_arch = ['[64,64]', '[128,128]', '[128,256]']
    agent_class = "a2c"
    num_steps = [300_000]
    features_dim = [256]
    policy = 'MlpPolicy'

    counter = 0
    for ms, na, d, lr, net, ns, fd in product(maze_size, num_agents, density, lr, net_arch, num_steps, features_dim):
        ac = agent_class

        run_string = f"-ms {ms} -na {na} -d {d} -lr {lr} -net {net} -ac {ac} -ns {ns} -fd {fd} -p {policy}"
        r = subprocess.run(['sbatch','train_marl.sh', run_string])
        counter +=1
        print(f"{counter}) {run_string} submitted")

    

if __name__ == "__main__":
    main()