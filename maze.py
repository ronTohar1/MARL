class maze:
    def __init__(self) -> None:
        self.W = "W"
        self.E = "-"
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.maze = []
        self.dim =  0
        self.num_of_agents = 0
        self.agents = []
        self.goals = []

    def generate_agents(self, num_of_agents):
        import random
        agents = []
        while len(agents) < num_of_agents:
            x = random.randint(0, self.dim-1)
            y = random.randint(0, self.dim-1)
            if (x,y) not in agents:
                agents.append((x,y))
        return agents
    
    def generate_goals(self, num_of_agents):
        import random
        goals = []
        while len(goals) < num_of_agents:
            x = random.randint(0, self.dim-1)
            y = random.randint(0, self.dim-1)
            if (x,y) not in goals and (x,y) not in self.agents:
                goals.append((x,y))
        return goals

    def check_paths(self, maze, agents, goals):
        import networkx as nx
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

    def generate_maze(self, percent_of_walls, tries=50):
        import random
        number_of_walls = int(percent_of_walls * self.dim * self.dim)
        
        # initialize the maze
        maze = [[0 for _ in range(self.dim)] for _ in range(self.dim)]

        # place the agents and goals in the maze
        for agent in self.agents:
            name = 'A' + str(self.agents.index(agent))
            maze[agent[0]][agent[1]] = name
        for goal in self.goals:
            name = 'G' + str(self.goals.index(goal))
            maze[goal[0]][goal[1]] = name

        # place the walls in the maze
        for _ in range(number_of_walls):
            for _ in range(tries):
                x = random.randint(0, self.dim-1)
                y = random.randint(0, self.dim-1)
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
    
    def reset(self, dim, num_of_agents, percent_of_walls):
        self.dim = dim
        self.num_of_agents = num_of_agents

        self.agents = self.generate_agents(num_of_agents)
        self.goals = self.generate_goals(num_of_agents)

        self.maze = self.generate_maze(percent_of_walls)

        print("Maze generated: ")
        for row in self.maze:
            print(row)
        print("Agents: ", self.agents)
        print("Goals: ", self.goals)

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


if __name__ == '__main__':
    m = maze()
    m.reset(10, 6, 0.6)
