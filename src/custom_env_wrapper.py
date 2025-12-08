import gymnasium as gym
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

    def reward(self, reward: float) -> float:
        """
        The reward parameter is the default reward
        """
        obs = self.env.unwrapped._get_obs()
        player_sum, dealer_card, usable_ace = obs
        
        if player_sum > 21:
            return -2.0 


        return reward
