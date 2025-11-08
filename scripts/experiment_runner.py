# scripts/experiment_runner.py
"""
Run multiple trainings with different seeds and (optionally) different environment reward parameters.
This script automates:
 - N seeds training
 - Evaluate each saved model and aggregate results

Note: For reward ablation (varying env reward weights), your env should accept a 'config' or kwargs to set those weights.
If your MultiP2PEnergyEnv has a parameter like fairness_lambda or carbon_penalty, you can modify the train code to accept/propagate it.
"""
import os
import subprocess
import argparse
from pathlib import Path

PYTHON = str(Path('myenv') / 'Scripts' / 'python.exe')

def run_train(seed, save_dir, extra_args=None):
    cmd = [PYTHON, '-m', 'train.train_ppo',
           '--n_agents', '6', '--episode_len', '24', '--n_envs', '4', '--total_timesteps', '200000',
           '--save_dir', save_dir, '--seed', str(seed)]
    if extra_args:
        cmd += extra_args
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

def run_evaluate(model_path, out_dir, n_agents=6, episode_len=24):
    cmd = [PYTHON, 'scripts/evaluate_model.py', 
           '--model', model_path, 
           '--n_episodes', '20', 
           '--save_dir', out_dir,
           '--n_agents', str(n_agents),
           '--episode_len', str(episode_len)]
    print('Evaluating:', ' '.join(cmd))
    subprocess.check_call(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--base_dir', type=str, default='experiments')
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    for s in range(args.n_seeds):
        sd = os.path.join(args.base_dir, f'seed_{s}')
        os.makedirs(sd, exist_ok=True)
        run_train(s, os.path.join(sd, 'models'))
        run_evaluate(os.path.join(sd, 'models', 'ppo_p2p.zip'), os.path.join(sd, 'results'), n_agents=6, episode_len=24)

    print('All seeds finished.')
