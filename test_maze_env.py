from maze_env import Maze, MazeConfig
import numpy as np
import networkx as nx
from tqdm import tqdm
def get_sizes():
    # return random int/float for each of the 3 sizes
    # dimension, num_agents, densirt
    return [np.random.randint(5, 10), np.random.randint(2,4), np.random.uniform(1, 1)]

def test_goals_are_free(num_of_envs=100):
    print("testing that initial env goals are free")
    for i in tqdm(range(num_of_envs)):
        env = Maze(*get_sizes())
        obs, info = env.reset()
        goals_cells = obs["goals"]
        for goal_cell in goals_cells:
            assert not env._is_obstacle(goal_cell)

def test_goals_are_always_free(num_of_envs=500, max_episode_steps=1000):
    print("testing that goals are always free thorughout the game")
    for i in tqdm(range(num_of_envs)):
        env = Maze(*get_sizes())
        obs, info = env.reset()
        # play a random game for up to max_episode_steps and assert that the goals are always free
        for step in range(max_episode_steps):
            # env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated , info = env.step(action)
            goals_cells = obs["goals"]
            agents_cells = obs["agents"]
            for i, goal_cell in enumerate(goals_cells):
                num_occasions = np.sum([tuple(agents_cells[i]) == tuple(goal_cell) for i in range(len(agents_cells))])
                condition = not env._is_obstacle(goal_cell) or num_occasions > 1 or (num_occasions == 1 and (not env.active_agents[i] or not tuple(agents_cells[i]) == tuple(goal_cell))) 
                assert condition, f"goal cell {goal_cell} is an obstacle! agent_pos={obs['agents'][i]}"
            if terminated or truncated:
                break

def game_will_end(num_of_envs=100, max_episode_steps=10000):
    print("Testing that game ends (max 10000) steps")
    for i in tqdm(range(num_of_envs)):
        env = Maze(*get_sizes())
        obs, info = env.reset()
        # play a random game for up to max_episode_steps and assert that the goals are always free
        for step in range(max_episode_steps):
            # env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated , info = env.step(action)
            if terminated or truncated:
                break
        assert step < max_episode_steps - 1, f"game did not end after {max_episode_steps} steps"

def test_rewards(num_of_envs=100, max_episode_steps=10000):
    print("Testing that rewards are correct")
    for i in tqdm(range(num_of_envs)):
        env = Maze(*get_sizes())
        obs, info = env.reset()
        # play a random game for up to max_episode_steps and assert that the goals are always free
        for step in range(max_episode_steps):
            # env.render()
            action = env.action_space.sample()
            obs, reward, terminated, truncated , info = env.step(action)
            if terminated or truncated:
                break
    
def main():
    # test_goals_are_always_free()
    # test_goals_are_free()
    game_will_end()

if __name__ == "__main__":
    main()