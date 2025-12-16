import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from blackjack_wrapper import BlackjackWrapper


def evaluate(model_path, env_id, episodes):
    filename = os.path.basename(model_path)
    run_name = os.path.splitext(filename)[0]
    model = DQN.load(model_path)
    env = BlackjackWrapper(gym.make(env_id, render_mode=None))

    log_path = os.path.join("logs", "evaluate", run_name)
    writer = SummaryWriter(log_dir=log_path)

    all_rewards = []
    wins = 0
    losses = 0
    draws = 0

    # Track hitting on soft 11 or less
    times_at_soft_11_or_less = 0
    times_hit_at_soft_11_or_less = 0

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        current_reward = 0
        episode_len = 0
        episode_result = None

        while not (done or truncated):
            player_sum, dealer_card, usable_ace = obs

            # Track if player is at 11 or less
            if player_sum <= 11:
                times_at_soft_11_or_less += 1
                action, _states = model.predict(obs, deterministic=True)
                # action 1 = hit, action 0 = stand
                if action == 1:
                    times_hit_at_soft_11_or_less += 1
            else:
                action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            current_reward += reward
            episode_len += 1

        all_rewards.append(current_reward)

        # Get actual game result from Blackjack environment
        # Blackjack-v1 returns: player_sum, dealer_card, usable_ace in obs
        # We need to check the raw environment for actual win/loss
        actual_result = env.unwrapped._get_obs()

        # Determine actual win/loss/draw from the episode result
        # In Blackjack: +1 = win, -1 = loss, 0 = draw
        if current_reward > 0:
            wins += 1
        elif current_reward < 0:
            losses += 1
        else:
            draws += 1

        writer.add_scalar("eval/episode_reward", current_reward, i)
        writer.add_scalar("eval/episode_length", episode_len, i)

    win_rate = (wins / episodes) * 100
    loss_rate = (losses / episodes) * 100
    draw_rate = (draws / episodes) * 100

    # Calculate hit rate on soft 11 or less
    hit_rate_soft_11_or_less = (times_hit_at_soft_11_or_less / times_at_soft_11_or_less * 100) if times_at_soft_11_or_less > 0 else 0

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    summary_text = (
        f"Mean Reward: {mean_reward:.3f} +/- {std_reward:.3f}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Loss Rate: {loss_rate:.1f}%\n"
        f"Draw Rate: {draw_rate:.1f}%\n"
        f"\nHit Rate on Sum ≤ 11: {hit_rate_soft_11_or_less:.1f}% ({times_hit_at_soft_11_or_less}/{times_at_soft_11_or_less})"
    )

    writer.add_text("eval/summary", summary_text, 0)

    env.close()
    writer.close()

if __name__ == '__main__':
    evaluate("results/tuning/DQN_gamma_0.1", "Blackjack-v1", episodes=1000)
