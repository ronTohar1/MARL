from typing import Optional
import gym
import numpy as np

class GridMaze(gym.Env):
    max_num_agents = 5

    def __init__(self, size, num_agents, density, map_name=None):
        self.size = size
        self.num_agents = num_agents
        self.density = density
        self.map_name = map_name

        self.observation_space = gym.spaces.Box(low=0, high=np.int8.max, shape=(self.size, self.size), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4 ** self.num_agents)
    
    def check_limits(self):
        if self.num_agents > GridMaze.max_num_agents:
            raise ValueError("size must be less than max_num_agents")
        

    def step(self, action):
        pass

    def reset(self, seed: Optional[int] = None):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

