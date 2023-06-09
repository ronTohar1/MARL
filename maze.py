import gym
import random
import networkx as nx



class Maze(gym.Env):
    def __init__(self, dim, num_of_agents, density = 0.2) -> None:
        
        Maze._check_legality(dim, num_of_agents, density)

        self.W = "W"
        self.E = "-"
        self.Agent = "A"
        self.Goal = "G"

        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.dim = dim
        self.num_of_agents = num_of_agents
        self.density = density

        self.maze = []
        self.agents = []
        self.goals = []
        self.graph = None


    def _check_legality(dim, num_agents, density):
        if dim < 2 or dim > 20:
            raise Exception("Dimension must be between 2 and 20")
        num_walls = int(density * dim * dim)
        if num_walls + num_agents*2 > dim * dim:
            raise Exception("Number of walls + Agents + Goals is greater than the dimension of the maze")
        
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

    def check_paths(self, maze, agents, goals):
        G = nx.Graph()
        for i in range(len(maze)):
            for j in range(len(maze)):
                if maze[i][j] != self.W:
                    G.add_node((i,j))
        for i in range(len(maze)):
            for j in range(len(maze)):
                if maze[i][j] != self.W:
                    if i > 0 and maze[i-1][j] != self.W:
                        G.add_edge((i,j), (i-1,j))
                    if i < len(maze)-1 and maze[i+1][j] != self.W:
                        G.add_edge((i,j), (i+1,j))
                    if j > 0 and maze[i][j-1] != self.W:
                        G.add_edge((i,j), (i,j-1))
                    if j < len(maze)-1 and maze[i][j+1] != self.W:
                        G.add_edge((i,j), (i,j+1))
        for i in range(len(agents)):
            if nx.has_path(G, agents[i], goals[i]) == False:
                return False
        return True


    
    def _get_agent_name(self, agent):
        return self.Agent + str(self.agents.index(agent))
    
    def _get_goal_name(self, goal):
        return self.Goal + str(self.goals.index(goal))
    
    def _reset_maze(self):
        self.maze = [[self.E for _ in range(self.dim)] for _ in range(self.dim)]
    
    def _add_agents(self, agents):
        for agent in agents:
            self.maze[agent[0]][agent[1]] = self._get_agent_name(agent) 

    def _add_goals(self, goals):
        for goal in goals:
            self.maze[goal[0]][goal[1]] = self._get_goal_name(goal)

    def _create_graph(self,):
        self.graph = nx.grid_graph(dim=(self.dim, self.dim))


    def _generate_maze(self, tries=50):
        if self.density < 0 or self.density > 0.5:
                    raise Exception("Percent of walls must be between 0 and 0.5")
        
        # reset the maze
        self._reset_maze()
        self.agents = self._generate_entities(self.num_of_agents)
        self._add_agents(self.agents)
        self.goals = self._generate_entities(self.num_of_agents)
        self._add_goals(self.goals)
        self._create_graph()
        
        number_of_walls = int(self.density * self.dim * self.dim)

        # place the walls in the maze
        no_more_empty_places = False
        for _ in range(number_of_walls):
            if no_more_empty_places:
                break
            for _ in range(tries):
                walls = self._generate_entities(1)
                if walls is None:
                    no_more_empty_places = True
                    break
                if maze[x][y] == 0:
                    maze[x][y] = self.W
                    if self.check_paths(maze, self.agents, self.goals) == False:
                        maze[x][y] = 0
                    else:
                        number_of_walls -= 1
                        break
        
        # place the empty spaces in the maze
        for i in range(self.dim):
            for j in range(self.dim):
                if maze[i][j] == 0:
                    maze[i][j] = self.E
        return maze

    def get_maze(self):
        return self.maze
    
    def reset(self, ):
        
        # self.agents = self.generate_agents(num_of_agents)
        # self.goals = self.generate_goals(num_of_agents)

        # self.maze = self.generate_maze(percent_of_walls)
        self.generate_maze()

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
    m = Maze()
    m.reset(10, 6, 0.6)
    total_time = time.time() - start_time
    print("Time taken to generate maze: ", total_time)
    m.render()
