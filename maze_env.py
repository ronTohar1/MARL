import gymnasium as gym
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



class MazeConfig:
    FREE = "f"
    OBSTACLE = "o"

    WALL_REP = "■"
    FREE_REP = "□"
    AGENT_REP = "A"
    GOAL_REP = "G"
    OBSTACLE_REP = "▢"

    STEP_REWARD = "step"
    # STAY_REWARD = "stay"
    WALL_REWARD = "hit_wall"
    COLLISION_REWARD = "collide"
    GOAL_REWARD = "goal"

    def __init__(self, dim, num_of_agents) -> None:
        self.dim = dim
        self.num_of_agents = num_of_agents
        self.maze = np.full((self.dim, self.dim), MazeConfig.FREE, dtype=object)

    def _get_empty_places(self, location_as_tuple=False):
        return [(i, j) for i,j in np.ndindex(self.dim, self.dim) if self.maze[i,j] == MazeConfig.FREE]
    
    def _place_obstacle(self, cell):
        self.maze[cell] = MazeConfig.OBSTACLE

    def _free_cell(self, cell):
        self.maze[cell] = MazeConfig.FREE



class MazeRepresentation:
    def __init__(self, maze_config) -> None:
        self.maze_config = maze_config
        self.maze = maze_config.maze
        self.agents = None
        self.goals = None
        self.graph = None
    

class Maze(gym.Env):
    def __init__(self, dim, num_of_agents, density = 0.2, max_steps=None, reward_config=None) -> None:
        
        Maze._check_legality(dim, num_of_agents, density)

        self.maze_config = MazeConfig(dim, num_of_agents)

        self.W = "■ "
        self.E = "□ "
        self.Agent = "A"
        self.Goal = "G"
        self.Blocking_Wall = "▢"

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.STAY = 4

        self.dim = dim
        self.num_of_agents = num_of_agents
        self.density = density

        self.maze = None
        self.graph = None

        self.agents_poitions = None
        self.goals_positions = None
        self.walls_positions = None
        self.active_agents = None
        self.terminated = None

        self.observation_space = gym.spaces.Dict({
            "agents": gym.spaces.Box(low=0, high=self.dim, shape=(self.num_of_agents, 2), dtype=np.int32),
            "goals": gym.spaces.Box(low=0, high=self.dim, shape=(self.num_of_agents, 2), dtype=np.int32),
            "walls": gym.spaces.MultiBinary(self.dim, self.dim),
        })
        self.action_space = gym.spaces.MultiDiscrete([5 for _ in range(self.num_of_agents)])
        self.reward_config = reward_config if reward_config is not None else {
            "step": -1,
            "stay": -1,
            "hit_wall": -10,
            "collide": -10,
            "goal": 100,
        }

        self.max_steps = max_steps
        self.curr_steps = 0


    


    def _check_legality(dim, num_agents, density):
        if dim < 2 or dim > 20:
            raise Exception("Dimension must be between 2 and 20")
        if num_agents*2 > dim * dim/2:
            raise Exception("Number of agents + goals is greater than half the dimension of the maze")
    
    def _get_free_cells(self, location_as_tuple=False):
        return [(i, j) for i,j in np.ndindex(self.dim, self.dim) if self.maze[i,j] == MazeConfig.FREE]
    
    def _is_obstacle(self, cell):
        return self.maze[cell] == MazeConfig.OBSTACLE

    # Generates X entities (can be anything to that matter - agents, goals, etc.)
    def _generate_entities(self, num_of_entities, obstacles=False):
        empty_cells = self._get_free_cells(location_as_tuple=False)
        if len(empty_cells) < num_of_entities:
            raise Exception(f"Not enough free cells to generate {num_of_entities} entities")
        
        entities_indices = random.sample(list(range(len(empty_cells))), num_of_entities)
        cells = [empty_cells[i] for i in entities_indices]
        
        if obstacles:
            for cell in cells:
                self.maze[cell] = MazeConfig.OBSTACLE
            
        return cells         
    
    def _reset_maze(self, seed=None):
        self.maze = np.full((self.dim, self.dim), MazeConfig.FREE)
        self.curr_steps = 0
        self.graph = self._init_graph()
        self.agents_poitions = self._generate_entities(self.num_of_agents, obstacles=True)
        self.goals_positions = self._generate_entities(self.num_of_agents)
        self.walls_positions = self._put_walls()
        self.active_agents = [self.agents_poitions[i] != self.goals_positions[i] for i in range(self.num_of_agents)]
        self.terminated = [False] * self.num_of_agents

    def _put_walls(self):
        num_of_walls = int(self.dim * self.dim * self.density)
        walls_put = 0
        free_cells = self._get_free_cells()
        walls_positions = []
        while len(walls_positions) < num_of_walls and len(free_cells) > 0:
            wall_cell = random.choice(free_cells)
            if self._add_wall(wall_cell):
                walls_positions.append(wall_cell)
            free_cells.remove(wall_cell)

        return walls_positions
        

    """ Tries to add wall and checks if there is still a path from each agent to its goal"""
    def _add_wall(self, wall):
        wall_edges = [(wall, neighbor) for neighbor in self.graph.neighbors(wall)]
        self.graph.remove_edges_from(wall_edges)
        if not self._agents_has_path_to_goal():
            self.graph.add_edges_from(wall_edges)
            return False
        
        self.maze[wall] = MazeConfig.OBSTACLE

        return True

    """ Checks if there is a path from each agent to its goal """
    def _agents_has_path_to_goal(self):
        graph = self.graph
        for agent_idx in range(self.num_of_agents):
            agent_cell = self.agents_poitions[agent_idx]
            goal_cell = self.goals_positions[agent_idx]

            # Add agent edges and check if there is path from agent to goal
            agent_edges = [(agent_cell, neighbor) for neighbor in self._get_adjacent_cells(agent_cell)]
            graph.add_edges_from(agent_edges)

            if not nx.has_path(self.graph, agent_cell, goal_cell):
                graph.remove_edges_from(agent_edges)
                return False
            
            graph.remove_edges_from(agent_edges)

        return True

    """ Creates the initial graph of the maze, after removing edges that go into or out of a wall or agent"""
    def _init_graph(self,):
        G = nx.grid_graph(dim=(self.dim, self.dim))
        edges_to_remove = [e for e in G.edges if self._is_obstacle(e[0]) or self._is_obstacle(e[1])]
        G.remove_edges_from(edges_to_remove)
        return G

    def _get_adjacent_cells(self, cell):
        adjacent_cells = [(cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)]
        adjacent_cells = [cell for cell in adjacent_cells if cell[0] >= 0 and cell[0] < self.dim and cell[1] >= 0 and cell[1] < self.dim]
        return adjacent_cells


    def _obs(self):
        walls_obs = np.zeros((self.dim, self.dim))
        for wall_pos in self.walls_positions:
            walls_obs[wall_pos] = 1
        observation = {
            "agents": np.array(self.agents_poitions),
            "goals": np.array(self.goals_positions),
            "walls": walls_obs,
        }
        return observation
    
    def _can_go_direction(self, agent_idx, direction):
        if direction == self.STAY:
            return True
        new_cell = self._get_new_cell(self.agents_poitions[agent_idx], direction)
        if not self._is_valid_cell(new_cell):
            return False
        if self._is_obstacle(new_cell):
            return False
        return True
        
        
    def _get_new_cell(self, cell, direction):
        if direction == self.UP:
            return (cell[0]-1, cell[1])
        elif direction == self.DOWN:
            return (cell[0]+1, cell[1])
        elif direction == self.LEFT:
            return (cell[0], cell[1]-1)
        elif direction == self.RIGHT:
            return (cell[0], cell[1]+1)
        elif direction == self.STAY:
            return cell
        else:
            raise ValueError("Invalid direction: {}".format(direction))

        
    def _is_valid_cell(self, cell):
        return cell[0] >= 0 and cell[0] < self.dim and cell[1] >= 0 and cell[1] < self.dim
        
    def _is_too_many_steps(self):
        if self.max_steps is None:
            return False
        return self.curr_steps >= self.max_steps
    
    def _move_agent(self, agent_idx, direction):
        print(f"Moving agent {agent_idx} in direction {direction}, status active: {self.active_agents[agent_idx]}")
        if not self.active_agents[agent_idx]:
            return 0
        
        new_cell = self._get_new_cell(self.agents_poitions[agent_idx], direction)
        if not self._can_go_direction(agent_idx, direction):
            if new_cell in self.agents_poitions:
                return self.reward_config[MazeConfig.COLLISION_REWARD]
            return self.reward_config[MazeConfig.WALL_REWARD]
        
        self.maze[self.agents_poitions[agent_idx]] = MazeConfig.FREE
        self.agents_poitions[agent_idx] = new_cell
        self.maze[self.agents_poitions[agent_idx]] = MazeConfig.OBSTACLE

        print(f"Agent {agent_idx} moved to {self.agents_poitions[agent_idx]}")

        if self.agents_poitions[agent_idx] == self.goals_positions[agent_idx]:
            self.active_agents[agent_idx] = False # agent done
            self.terminated[agent_idx] = True # mark agent as terminated
            self.maze[self.agents_poitions[agent_idx]] = MazeConfig.FREE # hide agent
            return self.reward_config[MazeConfig.GOAL_REWARD]
        
        return self.reward_config[MazeConfig.STEP_REWARD]

    def reset(self, seed=None):
        self._reset_maze(seed)
        return self._obs(), {}

    def step(self, action):
        # for each agent update the maze to move according to the action
        rewards = [0 for _ in range(self.num_of_agents)]
        num_active_agents = sum(self.active_agents)
        for agent_idx in range(self.num_of_agents):
            rewards[agent_idx] = self._move_agent(agent_idx, action[agent_idx])
            
        self.curr_steps += 1

        new_observation = self._obs()
        terminated = all(self.terminated)
        truncated = self._is_too_many_steps()
        reward = self._build_reward(rewards, num_active_agents)
        if terminated:
            print("done! total number of steps: {}".format(self.curr_steps))
        print(rewards)
        return new_observation, reward, terminated, truncated, {}

    def _build_reward(self, rewards, num_active_agents):
        # Create single reward from the rewards of each agent
        return sum(rewards) / num_active_agents

    def render(self):
        for i in range(self.dim):
            print_row = []
            for j in range(self.dim):
                cell = (i,j)
                cell_type = self.maze[i,j]
                if cell in self.agents_poitions and self.active_agents[self.agents_poitions.index(cell)]:
                        print_row.append(self.Agent + str(self.agents_poitions.index(cell)))
                elif cell in self.goals_positions and not(cell in self.agents_poitions and self.active_agents[self.agents_poitions.index(cell)]):
                    print_row.append(self.Goal + str(self.goals_positions.index(cell)))
                elif cell in self.walls_positions:
                    print_row.append(self.W)
                elif cell_type == MazeConfig.FREE:
                    print_row.append(self.E)
                else:
                    raise ValueError("Invalid cell rendered: {}".format(cell))
                    
            print("  ".join(print_row))

        for i in range(self.dim):
            print(" ".join(self.maze[i,:]))

        print()
