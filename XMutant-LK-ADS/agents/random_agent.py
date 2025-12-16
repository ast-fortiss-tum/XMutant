import gym
import numpy as np
from agents.agent import Agent


class RandomAgent(Agent):

    def __init__(self, env: gym.Env):
        super().__init__(env=env)

    def predict(self, obs: np.ndarray, speed: float = 0.0) -> np.ndarray:
        return self.env.action_space.sample()
