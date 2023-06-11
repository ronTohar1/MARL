import gym
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Maze(gym.Env):
    def __init__(self, dim, num_of_agents, density = 0.2) -> None:
        
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

        self.dim = dim
        self.num_of_agents = num_of_agents
        self.density = density

        self.maze = None
        self.agents = []
        self.goals = []
        self.graph = None

        self.observation_space = 


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
    
    def _get_goal_name(self, goal):
        return self.Goal + str(self.goals.index(goal))
    
    def _reset_maze(self):
        self.maze = np.full((self.dim, self.dim), self.E, dtype=object)
    
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

    """ Creates the initial graph of the maze, after removing edges that go into or out of a wall or agent"""
    def _create_initial_graph(self,):

        def is_wall(cell):
            return self.maze[cell] == self.W
        
        def is_agent(cell):
            return self.maze[cell].startswith(self.Agent)
        
        def is_wall_or_agent(cell):
            return is_wall(cell) or is_agent(cell)

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

            if not nx.has_path(self.graph, agent, goal):
                return False
            
            # remove agent edges from the graph
            self.graph.remove_edges_from(agent_edges)

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

    def _generate_maze(self, tries=50):
        # reset the maze
        self._reset_maze()
        self.agents = self._generate_entities(self.num_of_agents)
        self._add_agents()
        self.goals = self._generate_entities(self.num_of_agents)
        self._add_goals()
        self._create_initial_graph()
        
        number_of_walls = int(self.density * self.dim * self.dim)
        print("Number of walls to place: {}".format(number_of_walls))

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
       
    def get_maze(self):
        return self.maze
    
    def reset(self, ):
        
        # self.agents = self.generate_agents(num_of_agents)
        # self.goals = self.generate_goals(num_of_agents)

        # self.maze = self.generate_maze(percent_of_walls)
        self._generate_maze()

    def step(self, action):
        # for each agent update the maze to move according to the action
        for i in range(self.num_of_agents):
            agent = self.agents[i]
            goal = self.goals[i]
            if action[i] == self.UP and agent[0] > 0 and self.maze[agent[0]-1][agent[1]] != self.W:
                self.maze[agent[0]][agent[1]] = self.E
                self.maze[agent[0]-1][agent[1]] = 'A' + str(i)
                self.agents[i] = (agent[0]-1, agent[1])
            elif action[i] == self.DOWN and agent[0] < self.dim-1 and self.maze[agent[0]+1][agent[1]] != self.W:
                self.maze[agent[0]][agent[1]] = self.E
                self.maze[agent[0]+1][agent[1]] = 'A' + str(i)
                self.agents[i] = (agent[0]+1, agent[1])
            elif action[i] == self.LEFT and agent[1] > 0 and self.maze[agent[0]][agent[1]-1] != self.W:
                self.maze[agent[0]][agent[1]] = self.E
                self.maze[agent[0]][agent[1]-1] = 'A' + str(i)
                self.agents[i] = (agent[0], agent[1]-1)
            elif action[i] == self.RIGHT and agent[1] < self.dim-1 and self.maze[agent[0]][agent[1]+1] != self.W:
                self.maze[agent[0]][agent[1]] = self.E
                self.maze[agent[0]][agent[1]+1] = 'A' + str(i)
                self.agents[i] = (agent[0], agent[1]+1)
            else:
                pass

    def render(self):
        # print("Maze generated: ")
        for row in self.maze:
            print(row)
        # print("Agents: ", self.agents)
        # print("Goals: ", self.goals)


if __name__ == '__main__':
    import time
    start_time = time.time()
    m = Maze(10, 6, density=0.3)
    m.reset()
    total_time = time.time() - start_time
    print("Time taken to generate maze: ", total_time)
    m.render()
