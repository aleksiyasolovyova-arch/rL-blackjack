RL Blackjack Project - Run Instructions

This repository trains reinforcement learning agents on Gymnasium `Blackjack-v1` using Stable-Baselines3. It includes baseline training, a custom reward wrapper, a simple hyperparameter study, automatic evaluation, and TensorBoard logging.

1) Prerequisites
- Python 3.10+ recommended
- pip 23+
- Install PyTorch (CPU or CUDA) from https://pytorch.org/get-started/locally/
  Example (CPU):
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- Install other dependencies:
  ```
  pip install -r requirements.txt
  ```
- Optional virtual environment:
  ```
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

2) Project layout
- src/train_baseline.py : Train baseline agent
- src/train_custom_reward.py : Train with custom reward shaping
- src/train_extension.py : Hyperparameter study (e.g., gamma sweep)
- src/blackjack_wrapper.py : Observation wrapper
- src/custom_env_wrapper.py : Reward shaping wrapper
- src/evaluate.py : Evaluation, logs metrics to TensorBoard
- src/utils.py : Config loader
- config/*.yaml : Training configuration files
- logs/ : TensorBoard logs and checkpoints (created on first run)
- results/ : Saved models (created on first run)
- tensorboard.sh : Convenience script to start TensorBoard

3) Configuration
Training scripts read YAML files under `config/`.
Key fields: `environment` (e.g., `Blackjack-v1`), `algorithm` (`DQN`, `PPO`, `A2C`, `SAC`), `timesteps`, `checkpoint_freq`, `num_trials`.
For `train_extension.py`, `hyperparameter_study` contains `param_name` and `values` to sweep.
Edit the YAML you plan to use before running.

4) How to run (from project root)

Baseline training
```
python src/train_baseline.py
```
- Uses `config/config_baseline.yaml`
- Models: `results/baseline/<ALGO>_trialN`
- Logs: `logs/baseline/<ALGO>_trialN/`
- Checkpoints: `logs/baseline/<ALGO>_trialN/checkpoints/`
- Auto evaluation for 1000 episodes (logged to TensorBoard)

Custom reward shaping
```
python src/train_custom_reward.py
```
- Uses `config/config_custom.yaml`
- Saves to `results/custom/...` and logs to `logs/custom/...`
- Auto evaluation for 1000 episodes

Hyperparameter study (extension)
```
python src/train_extension.py
```
- Uses `config/config_extension.yaml`
- Sweeps `hyperparameter_study.values` for `param_name`
- Saves to `results/tuning/<ALGO>_<param>_<value>`
- Logs to `logs/tuning/<ALGO>_<param>_<value>/`
- Auto evaluation for 1000 episodes

5) TensorBoard
- Using helper script:
```
bash tensorboard.sh .
```
- Or directly:
```
tensorboard --logdir logs
```
Open the URL shown in the terminal (e.g., http://localhost:6006/).

6) Evaluation outputs
- After each training run, evaluation logs: episode reward and length, win/loss/draw rates, hit-rate when player sum is <= 11, and a summary text with mean/stdev.

7) Notes and tips
- Algorithms supported by training scripts: `DQN`, `PPO`, `A2C`, `SAC` (names must match SB3 classes). Evaluation defaults to DQN.
- Checkpoints are written under `logs/.../checkpoints/`; scripts do not auto-resume.
- No global random seeds are set; results vary between runs.
- GPU usage depends on your PyTorch installation (CUDA build) and algorithm support.

8) Quick start example
```
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/train_baseline.py
bash tensorboard.sh .
```

