from custom_env_wrapper import CustomRewardWrapper
from blackjack_wrapper import BlackjackWrapper
from stable_baselines3 import PPO, DQN, A2C, SAC
from evaluate import evaluate
from utils import load_config
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from pathlib import Path

def train_custom(cfg_path="config/config_custom.yaml"):
    os.chdir(Path(__file__).parent.parent.resolve())
    cfg = load_config(cfg_path)
    name = cfg['name']

    env = CustomRewardWrapper(BlackjackWrapper(gym.make(cfg['environment'], render_mode=None)),cfg)
    algo =  globals()[cfg['algorithm']]

    trial_name = f"{cfg['algorithm']}_{name}"
    log_dir = f"logs/custom/{trial_name}"
    checkpoint_dir = f"{log_dir}/checkpoints/"
    model_save_path = f"results/custom/{trial_name}"
    os.makedirs(log_dir, exist_ok=True)

    model = algo("MlpPolicy", env, verbose=1, tensorboard_log=f"{log_dir}/")
    cb = CheckpointCallback(save_freq=cfg["checkpoint_freq"],
                                save_path=checkpoint_dir,)
    model.learn(total_timesteps=cfg["timesteps"], callback=cb)
    model.save(model_save_path)

    evaluate(model_save_path, cfg['environment'], 1000)

if __name__ == '__main__': train_custom()
