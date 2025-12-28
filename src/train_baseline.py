from stable_baselines3 import PPO, DQN, A2C, SAC
import gymnasium as gym

from evaluate import evaluate
from blackjack_wrapper import BlackjackWrapper
from utils import load_config
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from pathlib import Path

def train_baseline(cfg_path="config/config_baseline.yaml"):

    os.chdir(Path(__file__).parent.parent.resolve())
    cfg = load_config(cfg_path)

    for trial in range(1, cfg['num_trials'] + 1):
        env =  BlackjackWrapper(gym.make(cfg['environment'], render_mode=None))
        algo =  globals()[cfg['algorithm']]

        trial_name = f"{cfg['algorithm']}_trial{trial}"
        log_dir = f"logs/baseline/{trial_name}"
        checkpoint_dir = f"{log_dir}/checkpoints/"
        model_save_path = f"results/baseline/{trial_name}"

        os.makedirs(log_dir, exist_ok=True)

        model = algo("MlpPolicy", env, verbose=1, tensorboard_log=f"{log_dir}/")

        cb = CheckpointCallback(save_freq=cfg["checkpoint_freq"],
                                save_path=checkpoint_dir)

        model.learn(total_timesteps=cfg["timesteps"], callback=cb)
        model.save(model_save_path)

        evaluate(model_save_path, cfg['environment'], 1000)

if __name__ == '__main__': train_baseline()
