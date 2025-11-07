#!/usr/bin/env bash
# run_experiment.sh
python3 train/train_ppo.py --n_agents 6 --episode_len 24 --n_envs 4 --total_timesteps 200000 --save_dir models
