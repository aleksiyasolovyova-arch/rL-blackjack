import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    """
    Research-based reward function for Blackjack incorporating:
    - Basic strategy principles from Baldwin et al. (1956)
    - Expected value considerations for different hand situations
    - Graduated penalties for poor decisions
    - Bonus rewards for navigating dangerous situations successfully
    """

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg
        self.previous_obs = None

    def reward(self, reward: float) -> float:
        """
        Enhanced reward function based on blackjack research.

        Key insights from research:
        - Hard 16 is the worst hand (highest expected loss)
        - Dealer upcards 9, 10, Ace are strongest
        - Dealer upcards 4, 5, 6 are weakest (highest bust probability)
        - Soft hands provide flexibility and lower risk
        """
        obs = self.env.unwrapped._get_obs()
        player_sum, dealer_card, usable_ace = obs

        action_was_hit = self.previous_obs is not None and player_sum != self.previous_obs[0]

        self.previous_obs = obs

        # Base reward from environment (1, 0, -1 for win/draw/loss)
        shaped_reward = reward

        # Severe penalty for busting (research shows this is the main way players lose)
        if player_sum > 21:
            return -2.5

        # === WIN/LOSS MODIFIERS ===
        if reward != 0:  # Episode ended (win or loss)

            # DANGER ZONE BONUS: Successfully navigating hard 16
            # Research shows hard 16 is the worst hand with highest expected loss
            # Reward agent for surviving these situations
            if self.previous_obs is not None:
                prev_sum = self.previous_obs[0]
                prev_dealer = self.previous_obs[1]
                prev_soft = self.previous_obs[2]

                # Hard 16 against strong dealer cards (9, 10, Ace)
                if prev_sum == 16 and not prev_soft and prev_dealer in [9, 10, 1]:
                    if reward > 0:  # Won from terrible position
                        shaped_reward += 0.3
                    elif reward < 0:  # Lost from terrible position (expected)
                        shaped_reward += 0.1  # Small consolation for expected loss

                # Hard 12-15 against strong dealer cards
                elif prev_sum in [12, 13, 14, 15] and not prev_soft and prev_dealer in [9, 10, 1]:
                    if reward > 0:
                        shaped_reward += 0.2

                # Successfully used soft hand flexibility
                elif prev_soft and reward > 0:
                    shaped_reward += 0.15

        # Small rewards/penalties during play to guide learning
        else:

            dealer_strength = self._get_dealer_strength(dealer_card)

            # Strong position: High hand value, especially vs weak dealer
            if player_sum >= 17 and player_sum <= 21:
                if dealer_strength == "weak":
                    shaped_reward += 0.15  # Great position
                elif dealer_strength == "strong":
                    shaped_reward += 0.05  # Decent but dealer has advantage

            # Research emphasizes soft hands are valuable due to flexibility
            if usable_ace and player_sum >= 18:
                shaped_reward += 0.1

            # Small penalty for being in statistically worst positions
            if not usable_ace:  # Hard hand
                if player_sum == 16:
                    # Worst hand in blackjack according to research
                    if dealer_card in [9, 10, 1]:  # Against strong dealer
                        shaped_reward -= 0.15
                    else:
                        shaped_reward -= 0.05

                elif player_sum in [12, 13, 14, 15]:
                    # Other difficult hands
                    if dealer_card in [9, 10, 1]:
                        shaped_reward -= 0.1
                    elif dealer_card in [4, 5, 6]:
                        # Actually good position - dealer likely to bust
                        shaped_reward += 0.05

            # Reward reaching strong positions
            if player_sum == 20 or player_sum == 21:
                shaped_reward += 0.2
            elif player_sum == 19:
                shaped_reward += 0.1

            # Penalty for standing on weak totals (going against basic strategy)
            if player_sum <= 11:
                shaped_reward -= 0.05

        # Clip to reasonable range
        shaped_reward = np.clip(shaped_reward, -3.0, 2.0)

        return shaped_reward

    def _get_dealer_strength(self, dealer_card: int) -> str:
        """
        Categorize dealer upcard strength based on research.

        Research shows:
        - 2-6: Dealer has highest bust probability (weak)
        - 7-8: Medium strength
        - 9, 10, Ace: Strongest (lowest bust probability)
        """
        if dealer_card in [4, 5, 6]:
            return "weak"
        elif dealer_card in [2, 3]:
            return "medium_weak"
        elif dealer_card in [7, 8]:
            return "medium"
        else:  # 9, 10, Ace (1)
            return "strong"

    def reset(self, **kwargs):
        """Reset the previous observation tracking."""
        self.previous_obs = None
        return self.env.reset(**kwargs)