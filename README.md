## First try with DQN:
Has an **ep_rew_mean** of -0.16, -0.13, -0.2  per trial. What this means is that the agent has learned to play, but it is not perfect. In practice this means it’s definitely better than random guessing, but hasn’t mastered the rules(like exactly when to hit on a “soft 17”)

**For report**:
The baseline DQN algorithm successfully converged to a strategy quite better than a random play, reducing the average loss per hand from a theoretical random baseline of -0.4 to approximately -0.16.

Has an **ep_len_mean** of 1.65, 1.63 and 1.71 per trial. What this proves is that the agent is taking risks instead of “sticking” immediately on every hand. With a score of approximately 1.66, it implies that in roughly 66% of games, the agent asks for at least one extra card.