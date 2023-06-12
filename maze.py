import gymnasium as gym
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Maze(gym.Env):
    def __init__(self, dim, num_of_agents, density = 0.2, max_steps=None, reward_config=None) -> None:
        
        Maze._check_legality(dim, num_of_agents, density)

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
        self.agents = []
        self.active_agents = []
        self.goals = []
        self.graph = None

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim, self.dim), dtype=np.int32)
        # action space is a box from 0 to 4 (up, down, left, right, stay) for each agent
        # self.action_space = gym.spaces.Box(low=0, high=4, shape=(self.num_of_agents,), dtype=np.int32)
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


    STEP = "step"
    STAY = "stay"
    HIT_WALL = "hit_wall"
    COLLIDE = "collide"
    GOAL = "goal"


    def _check_legality(dim, num_agents, density):
        if dim < 2 or dim > 20:
            raise Exception("Dimension must be between 2 and 20")
        if num_agents*2 > dim * dim/2:
            raise Exception("Number of agents + goals is greater than half the dimension of the maze")
        # num_walls = int(density * dim * dim)
        # if num_walls + num_agents*2 > dim * dim:
            # raise Exception("Number of walls + Agents + Goals is greater than the dimension of the maze")
        
    def _draw_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()
    
    def _get_empty_places(self, location_as_tuple=False):
        if location_as_tuple:
            return [(i,j) for i in range(self.dim) for j in range(self.dim) if self.maze[i][j] == self.E]
        return [i*self.dim + j for i in range(self.dim) for j in range(self.dim) if self.maze[i][j] == self.E]
    
    def _convert_location_to_tuple(self, place):
        return (place // self.dim, place % self.dim)

    # Generates X entities (can be anything to that matter - agents, goals, etc.)
    def _generate_entities(self, num_of_entities):

        empty_places = self._get_empty_places(location_as_tuple=False)
        if len(empty_places) < num_of_entities:
            return None
        
        entities = random.sample(empty_places, num_of_entities)
        entities = [self._convert_location_to_tuple(entity) for entity in entities]
        return entities         
    
    def _get_agent_name(self, agent):
        return self.Agent + str(self.agents.index(agent))
    
    def _get_agent_name_by_index(self, index):
        return self.Agent + str(index)
    
    def _get_goal_name(self, goal):
        return self.Goal + str(self.goals.index(goal))
    
    def _get_goal_name_by_index(self, index):
        return self.Goal + str(index)
    
    def _reset_maze(self):
        self.maze = np.full((self.dim, self.dim), self.E, dtype=object)
        self.agents = None
        self.goals = None
        self.graph = None
        self.curr_steps = 0
    
    def _add_agents(self):
        if self.agents is None:
            raise Exception("Agents are not initialized")
        for agent in self.agents:
            name = self._get_agent_name(agent) 
            self.maze[agent] = name

    def _add_goals(self):
        if self.goals is None:
            raise Exception("Goals are not initialized")
        for goal in self.goals:
            self.maze[goal] = self._get_goal_name(goal)

    def _is_wall(self, cell):
            return self.maze[cell] == self.W
        
    def _is_agent(self, cell):
        return self.maze[cell].startswith(self.Agent)
    
    def _is_goal(self, cell):
        return cell in self.goals
    
    def _is_goal_of_agent(self, cell, agent):
        return cell in self.goals and self.goals.index(cell) == self.agents.index(agent)
        
    """ Creates the initial graph of the maze, after removing edges that go into or out of a wall or agent"""
    def _create_initial_graph(self,):

        def is_wall_or_agent(cell):
            return self._is_wall(cell) or self._is_agent(cell)

        self.graph = nx.grid_graph(dim=(self.dim, self.dim))

        # Remove all edges that go into or out of a wall or agent
        edges_to_remove = [e for e in self.graph.edges if is_wall_or_agent(e[0]) or is_wall_or_agent(e[1])]
        self.graph.remove_edges_from(edges_to_remove)

    def _get_adjacent_cells(self, cell):
        adjacent_cells = [(cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)]
        adjacent_cells = [cell for cell in adjacent_cells if cell[0] >= 0 and cell[0] < self.dim and cell[1] >= 0 and cell[1] < self.dim]
        return adjacent_cells

    """ Checks if there is a path from each agent to its goal """
    def _agents_has_path_to_goal(self):
        for agent in self.agents:
            # add agent edges to the graph
            agent_edges = [(agent, neighbor) for neighbor in self._get_adjacent_cells(agent)]
            self.graph.add_edges_from(agent_edges)
            goal = self.goals[self.agents.index(agent)]

            has_path = True
            if not nx.has_path(self.graph, agent, goal):
                has_path = False
            
            # remove agent edges from the graph
            self.graph.remove_edges_from(agent_edges)
            if not has_path:
                return False
            

        return True

    """ Tries to add wall and checks if there is still a path from each agent to its goal"""
    def _add_wall(self, wall):
        if self.maze[wall] != self.E:
            return False
        
        # remove wall edges from the graph
        wall_edges = [(wall, neighbor) for neighbor in self.graph.neighbors(wall)]
        self.graph.remove_edges_from(wall_edges)
        if not self._agents_has_path_to_goal():
            # add wall edges to the graph
            self.graph.add_edges_from(wall_edges)
            self.maze[wall] = self.Blocking_Wall # mark the wall as a place that cannot be a wall
            return False
        
        # add wall to the maze
        self.maze[wall] = self.W

        return True
    
    def _is_agent_active(self, agent_index):
        return self.active_agents[agent_index]

    def _init_agent_active_list(self):
        self.active_agents = [self.agents[i] != self.goals[i] for i in range(self.num_of_agents)]

    def _generate_maze(self, tries=50):
        # reset the maze
        self._reset_maze()
        self.agents = self._generate_entities(self.num_of_agents)
        self._add_agents()
        self.goals = self._generate_entities(self.num_of_agents)
        self._add_goals()
        self._create_initial_graph()
        self._init_agent_active_list()
        
        number_of_walls = int(self.density * self.dim * self.dim)

        # place the walls in the maze
        placed_walls = 0
        while placed_walls < number_of_walls:
                walls = self._generate_entities(1)
                if walls is None: # No more places to place a wall
                    print("Cannot place more walls, maze is full\nPlaced {} walls out of {}".format(placed_walls, number_of_walls))
                    break
                wall = walls[0]
                if self._add_wall(wall):
                    placed_walls += 1

        # Remove all "Blocking_Wall" from the maze
        self.maze[self.maze == self.Blocking_Wall] = self.E
       
    def get_maze(self):
        return self.maze
    
    def agent_obs(self, agent_name):
        return int(agent_name[len(self.Agent):])
        
    def goal_obs(self, goal_name):
        return int(goal_name[len(self.Goal):])
    
    def _get_cell_obs(self, cell):
        # 0 for empty cell, 1 for wall, 2 + index for agent or goal
        if cell in self.agents:
            return 2 + self.agent_obs(self._get_agent_name(cell))
        elif cell in self.goals:
            return 2 + self.goal_obs(self._get_goal_name(cell))
        elif self.maze[cell] == self.W:
            return 1
        elif self.maze[cell] == self.E:
            return 0
        else:
            raise Exception("Unknown cell type")
        

    def _get_observation(self):
        observation = [[self._get_cell_obs((i,j)) for i in range(self.dim)] for j in range(self.dim)]
        return observation
    
    def _can_go_direction(self, agent, direction):
        if direction == self.UP:
            new_cell = (agent[0]-1, agent[1])
            return agent[0] > 0 and self.maze[agent[0]-1][agent[1]] != self.W and not (new_cell in self.agents and self._is_agent_active(self.agents.index(new_cell)))
        elif direction == self.DOWN:
            new_cell = (agent[0]+1, agent[1])
            return agent[0] < self.dim-1 and self.maze[agent[0]+1][agent[1]] != self.W and not (new_cell in self.agents and self._is_agent_active(self.agents.index(new_cell)))
        elif direction == self.LEFT:
            new_cell = (agent[0], agent[1]-1)
            return agent[1] > 0 and self.maze[agent[0]][agent[1]-1] != self.W and not (new_cell in self.agents and self._is_agent_active(self.agents.index(new_cell)))
        elif self.RIGHT:
            new_cell = (agent[0], agent[1]+1)
            return agent[1] < self.dim-1 and self.maze[agent[0]][agent[1]+1] != self.W and not (new_cell in self.agents and self._is_agent_active(self.agents.index(new_cell)))
        
    def _get_new_position(self, agent, direction):
        if direction == self.UP:
            return (agent[0]-1, agent[1])
        elif direction == self.DOWN:
            return (agent[0]+1, agent[1])
        elif direction == self.LEFT:
            return (agent[0], agent[1]-1)
        elif direction == self.RIGHT:
            return (agent[0], agent[1]+1)
        elif direction == self.STAY:
            return agent
        else:
            raise ValueError("Invalid direction: {}".format(direction))

        
    def _is_valid_cell(self, cell):
        return cell[0] >= 0 and cell[0] < self.dim and cell[1] >= 0 and cell[1] < self.dim
        
    def _is_done(self):
        # all agent are at their goals positions
        print(self.agents, self.goals)
        return all([self.agents[i] == self.goals[i] for i in range(self.num_of_agents)])


    def _is_too_many_steps(self):
        if self.max_steps is None:
            return False
        return self.curr_steps >= self.max_steps

    def reset(self, seed=None):
        self._generate_maze()
        return self._get_observation(), {}

    def step(self, action):
        # for each agent update the maze to move according to the action
        rewards = [0 for _ in range(self.num_of_agents)]
        for i in range(self.num_of_agents):
            if not self._is_agent_active(i):
                rewards[i] = 0 if action[i] == self.STAY else self.reward_config[Maze.STEP]    
                continue
            agent = self.agents[i]
            new_position = self._get_new_position(agent, action[i])
            if self._can_go_direction(agent, action[i]):
                self.maze[agent] = self.E if not self._is_goal(agent) else self._get_goal_name(agent)
                self.agents[i] = new_position
                agent = self.agents[i]
                self.maze[self.agents[i]] = self._get_agent_name_by_index(i)

                if self._is_goal_of_agent(new_position, agent):
                    rewards[i] = self.reward_config[Maze.GOAL]
                    self.active_agents[i] = False

                else:
                    rewards[i] = self.reward_config[Maze.STEP]

            elif self._is_valid_cell(new_position) and self._is_agent(new_position):
                rewards[i] = self.reward_config[Maze.HIT_WALL]
            else:
                rewards[i] = self.reward_config[Maze.COLLIDE]
            
        self.curr_steps += 1

        new_observation = self._get_observation()
        terminated = self._is_done()
        truncated = self._is_too_many_steps()
        reward = self._build_reward(rewards)
        if terminated:
            print("done! total number of steps: {}".format(self.curr_steps))
        return new_observation, reward, terminated, truncated, {}

    def _build_reward(self, rewards):
        # Create single reward from the rewards of each agent
        return sum(rewards) / self.num_of_agents

    def render(self):
        for row in self.maze:
            print(row)
