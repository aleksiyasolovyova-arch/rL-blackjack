import os

from custom_env_wrapper import CustomRewardWrapper
from stable_baselines3 import DQN
from evaluate import evaluate
from blackjack_wrapper import BlackjackWrapper
from utils import load_config
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path

def train_extension(cfg_path="config/config_extension.yaml"):
    os.chdir(Path(__file__).parent.parent.resolve())
    cfg = load_config(cfg_path)

    param = cfg['hyperparameter_study']['param_name']
    values = cfg['hyperparameter_study']['values']

    for value in values:
        env = CustomRewardWrapper(BlackjackWrapper(gym.make(cfg['environment'], render_mode=None)),cfg)
        run_name = f"{cfg['algorithm']}_{param}_{value}"
        model_save_path = f"results/tuning/{run_name}"
        log_path = f"logs/tuning/{run_name}/"
        checkpoint_path = f"logs/tuning/{run_name}/checkpoints/"

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=checkpoint_path,
            name_prefix=f"rl_model_{run_name}"
        )

        model_kwargs = {param: value}

        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
            **model_kwargs
        )

        model.learn(total_timesteps=cfg["timesteps"], callback=checkpoint_callback)
        model.save(model_save_path)
        env.close()

        evaluate(model_save_path, cfg['environment'], 1000)


if __name__ == '__main__': train_extension()
