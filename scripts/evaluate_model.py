# scripts/evaluate_model.py
"""
Evaluating a saved PPO model across multiple episodes and saving metrics + plots.
"""
import argparse
import os
import sys
import csv
import numpy as np

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.multi_p2p_env import MultiP2PEnergyEnv

def evaluate(model_path, n_episodes, save_dir, n_agents, episode_len, deterministic=True):
    os.makedirs(save_dir, exist_ok=True)
    model = PPO.load(model_path)
    results = []
    for ep in range(n_episodes):
        env = MultiP2PEnergyEnv(n_agents=n_agents, episode_len=episode_len, seed=ep)
        obs, info = env.reset()
        ep_rewards = []
        ep_profits = np.zeros(n_agents)
        ep_emissions = np.zeros(n_agents)
        ep_ginis = []
        t = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(float(reward))
            profits = info.get('profits')
            emissions = info.get('emissions')
            gini = info.get('gini')
            if profits is not None:
                ep_profits += np.array(profits)
            if emissions is not None:
                ep_emissions += np.array(emissions)
            if gini is not None:
                ep_ginis.append(float(gini))
            t += 1
            if t > episode_len + 5:
                break
        results.append({
            'episode': ep,
            'total_reward': float(np.sum(ep_rewards)),
            'mean_step_reward': float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            'total_profit_sum': float(np.sum(ep_profits)),
            'mean_profit_per_agent': float(np.mean(ep_profits)),
            'total_emissions': float(np.sum(ep_emissions)),
            'mean_gini': float(np.mean(ep_ginis)) if ep_ginis else None
        })
        print(f"Ep {ep}: total_reward={results[-1]['total_reward']:.4f}, profit_sum={results[-1]['total_profit_sum']:.4f}, mean_gini={results[-1]['mean_gini']}")

    # save CSV
    csv_path = os.path.join(save_dir, 'evaluation.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Saved CSV:", csv_path)

    # plots
    rewards = [r['total_reward'] for r in results]
    profits = [r['total_profit_sum'] for r in results]
    ginis = [r['mean_gini'] for r in results]

    plt.figure()
    plt.plot(rewards, marker='o', linestyle='-')
    plt.title('Total reward per episode')
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'rewards_per_episode.png'))

    plt.figure()
    plt.plot(profits, marker='o', linestyle='-')
    plt.title('Total profit sum per episode')
    plt.xlabel('episode')
    plt.ylabel('profit_sum')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'profits_per_episode.png'))

    if any(g is not None for g in ginis):
        plt.figure()
        plt.plot(ginis, marker='o', linestyle='-')
        plt.title('Mean Gini per episode')
        plt.xlabel('episode')
        plt.ylabel('gini')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'gini_per_episode.png'))

    print("Saved plots to", save_dir)
    return csv_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--n_agents', type=int, default=6)
    parser.add_argument('--episode_len', type=int, default=24)
    parser.add_argument('--deterministic', action='store_true', default=True)
    args = parser.parse_args()
    evaluate(args.model, args.n_episodes, args.save_dir, args.n_agents, args.episode_len, deterministic=args.deterministic)
