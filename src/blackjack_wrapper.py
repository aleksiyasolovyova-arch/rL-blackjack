import gymnasium as gym
import numpy as np
import os

from gymnasium.core import ObsType, WrapperObsType
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import load_config

class BlackjackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=32, shape=(3,), dtype=np.float32)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return np.array(observation, dtype=np.float32)
