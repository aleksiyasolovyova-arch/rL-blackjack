import gymnasium as gym
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg
        self.last_action = None

    def step(self, action):
        self.last_action = action  # Capture action
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward(reward)  # RewardWrapper will call this
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)

    def reward(self, reward: float) -> float:
        obs = self.env.unwrapped._get_obs()
        player_sum, dealer_card, usable_ace = obs

        # Bust penalty
        if player_sum > 21:
            return -2.0

        # Small shaping for obvious decisions (avoid over-shaping)
        if not usable_ace and 12 <= player_sum <= 16:
            # Dealer bust cards (4-6) -> encourage standing
            if dealer_card in [4, 5, 6] and self.step == 0:
                reward += 0.1
            # Dealer strong cards (7-A) -> encourage hitting
            elif dealer_card >= 7 and self.last_action == 1:
                reward += 0.1

        return reward
