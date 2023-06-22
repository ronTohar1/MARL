from pogema import pogema_v0, Hard8x8
from pogema.grid_config import GridConfig
import gym
from pogema.integrations.make_pogema import pogema_v0
from stable_baselines3 import PPO


env = gym.make("Pogema-v0", grid_config=GridConfig(integration='gym',num_agents=4, size=10))
# obs, info = env.reset(), None

# while True:
#     # Using random policy to make actions
#     # action = env.sample_actions()
#     action = env.action_space.sample()
#     print("Taking action: ", action)
#     print("Info: ", info)
#     print("Observation: ", obs)
#     obs, reward, done, info = env.step(action)

#     env.render()
#     # if done:
#         # break

agent = PPO("MlpPolicy", env, verbose=2)
agent.learn(total_timesteps=10000)
