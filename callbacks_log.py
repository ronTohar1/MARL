from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.num_steps_per_episode = 0

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # print("Rewards of step = ", self.locals["rewards"])
        value = self.locals["rewards"][0]
        print("rewards is" , self.locals["rewards"])
        self.logger.record("episode/reward", value)
        self.rewards.append(value)
        self.num_steps_per_episode += 1
        return True
    
    def _on_rollout_end(self) -> bool:
        # # log the mean reward of the rollout
        # # print("Rewards of rollout = ", self.locals["rewards"])
        # value = np.mean(self.rewards)
        # self.logger.record("episode/mean_reward", value)
        # self.logger.record("episode/length", self.num_steps_per_episode)
        # self.rewards = []
        # self.num_steps_per_episode = 0
        return True
    
    