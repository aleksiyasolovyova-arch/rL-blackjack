from custom_env_wrapper import CustomRewardWrapper
from blackjack_wrapper import BlackjackWrapper
from stable_baselines3 import PPO, DQN, A2C, SAC
from utils import load_config
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import os

def train_custom(cfg_path="../config/config_custom.yaml"):
    cfg = load_config(cfg_path)
    name = cfg['name']
    env = CustomRewardWrapper(BlackjackWrapper(gym.make(cfg['environment'], render_mode=None)),cfg)
    algo =  globals()[cfg['algorithm']]

    os.makedirs(f"results/custom/{cfg['algorithm']}_{name}", exist_ok=True)
    os.makedirs(f"logs/custom/{cfg['algorithm']}_{name}", exist_ok=True)

    model = algo("MlpPolicy", env, verbose=1, tensorboard_log=f"logs/custom//{cfg['algorithm']}_{name}/")
    cb = CheckpointCallback(save_freq=cfg["checkpoint_freq"],
                                save_path=f"logs/custom/{cfg['algorithm']}_{name}/checkpoints/")
    model.learn(total_timesteps=cfg["timesteps"], callback=cb)
    model.save(f"results/custom/{cfg['algorithm']}_{name}")

if __name__ == '__main__': train_custom()
