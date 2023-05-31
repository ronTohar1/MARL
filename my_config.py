from pogema import GridConfig, Hard32x32

map_size = 8
# Define hard 16x16 grid
grid_config = GridConfig(num_agents=2,  # number of agents
                         size=map_size, # size of the grid
                         density=0.3,  # obstacle density
                         seed=1,  # set to None for random 
                                  # obstacles, agents and targets 
                                  # positions at each reset
                         max_episode_steps=128,  # horizon
                         observation_type="MAPF",  # observation type - centralized MAPF
                         obs_radius=map_size,  # observation radius - 
                         )