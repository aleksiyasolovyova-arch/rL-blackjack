import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.blackjack_wrapper import BlackjackWrapper


def evaluate(model_path, env_id, episodes):
    filename = os.path.basename(model_path)
    run_name = os.path.splitext(filename)[0]
    model =  DQN.load(model_path)
    env = BlackjackWrapper(gym.make(env_id, render_mode=None))

    log_path = os.path.join("logs", "evaluate", run_name)
    writer = SummaryWriter(log_dir=log_path)

    all_rewards = []

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        current_reward = 0
        episode_len = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            current_reward += reward
            episode_len += 1
        all_rewards.append(current_reward)
        writer.add_scalar("eval/episode_reward", current_reward, i)
        writer.add_scalar("eval/episode_length", episode_len, i)

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    writer.add_text("eval/summary", f"Mean Reward: {mean_reward:.3f} +/- {std_reward:.3f}", 0)

    env.close()
    writer.close()

if __name__ == '__main__':
    evaluate("results/baseline/DQN_trial1.zip", "Blackjack-v1", episodes=1000)