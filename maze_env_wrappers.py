from maze_env import Maze
import gymnasium
import numpy as np

class MazeImageWrapper(gymnasium.Env):

    def __init__(self, env: Maze):
        self.env = env
        self.num_channels = env.num_of_agents + 1
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(self.num_channels, env.dim, env.dim), dtype=np.uint8)
        self.action_space = env.action_space
        # self.action_names = env.action_names

    def _modify_obs(self, obs):
        agents_positions = obs["agents"]
        goals_positions = obs["goals"]
        walls_channel = obs["walls"]

        num_agents = len(agents_positions)
        channels = np.zeros((self.num_channels, self.env.dim, self.env.dim), dtype=np.uint8)
        for agent_idx in range(num_agents):
            agent_position = agents_positions[agent_idx]
            goal_position = goals_positions[agent_idx]
            channels[agent_idx, agent_position[0], agent_position[1]] = 1
            channels[agent_idx, goal_position[0], goal_position[1]] = 255

        channels[-1] = walls_channel
        return channels



    def step(self, action):
        obs, reward, terminated, truncated , info = self.env.step(action)
        obs = self._modify_obs(obs)
        return obs, reward, terminated, truncated , info

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self._modify_obs(obs)
        return obs, info
    
    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

