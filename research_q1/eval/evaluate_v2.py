"""
evaluate_v2.py — Pure Torch Actor Evaluation for SLIM v2
========================================================

This script avoids SB3's RecursionError by extracting only the actor 
weights and running them in a vanilla Torch nn.Module. 
NO SB3 PPO CLASS IS USED AT RUNTIME.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import zipfile
import io

# ─── Path fix ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

# ─── Vanilla Torch Actor ──────────────────────────────────────────────────────
class PureTorchActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # SB3 PPO default: 2 layers of 256
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.action_net = nn.Linear(256, output_dim)
        
    def forward(self, x):
        features = self.policy_net(x)
        return self.action_net(features)

def load_pure_policy(zip_path, input_dim, output_dim):
    actor = PureTorchActor(input_dim, output_dim)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        with archive.open('policy.pth') as file:
            state_dict = torch.load(file, map_location='cpu')
            
    # Map SB3 keys to our vanilla model
    # SB3: mlp_extractor.policy_net.0.weight -> policy_net.0.weight
    # SB3: mlp_extractor.policy_net.2.weight -> policy_net.2.weight
    new_state = {}
    new_state['policy_net.0.weight'] = state_dict['mlp_extractor.policy_net.0.weight']
    new_state['policy_net.0.bias']   = state_dict['mlp_extractor.policy_net.0.bias']
    new_state['policy_net.2.weight'] = state_dict['mlp_extractor.policy_net.2.weight']
    new_state['policy_net.2.bias']   = state_dict['mlp_extractor.policy_net.2.bias']
    new_state['action_net.weight']   = state_dict['action_net.weight']
    new_state['action_net.bias']     = state_dict['action_net.bias']
    
    actor.load_state_dict(new_state)
    actor.eval()
    return actor

# ─── Evaluation Main ─────────────────────────────────────────────────────────
EVAL_SEEDS = [0, 7, 13, 21, 42]

def evaluate_v2(model_path, n_agents, episodes):
    print("\n" + "="*60)
    print("  SLIM v2 EVALUATION (Pure Torch Actor Isolation)")
    print("="*60)

    # 1. Load weights
    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    # obs_dim = 105, action_dim = n_agents * 2 = 8
    actor = load_pure_policy(zip_path, 105, 8)
    print("  ✓ Pure Torch Actor loaded (no SB3).")

    # 2. Load Normalization
    norm_path = zip_path.replace("slim_v2_final.zip", "vec_normalize_v2.pkl")
    with open(norm_path, "rb") as f:
        vn_data = pickle.load(f)
        mean, var = vn_data.obs_rms.mean, vn_data.obs_rms.var

    # 3. Eval Loop
    results = []
    for seed in EVAL_SEEDS[:episodes]:
        env = EnergyMarketEnvSLIM(n_agents=4, data_file="processed_hybrid_data.csv", random_start_day=True, seed=seed)
        obs, _ = env.reset(seed=seed)
        
        ep_reward, ep_p2p, n_buyers_h = 0.0, 0.0, []
        with torch.no_grad():
            for _ in range(168):
                norm_obs = (obs - mean) / np.sqrt(var + 1e-8)
                norm_obs = np.clip(norm_obs, -10.0, 10.0)
                obs_t = torch.as_tensor(norm_obs[None, :], dtype=torch.float32)
                
                action = actor(obs_t).numpy()[0]
                
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_p2p += info.get('market/p2p_volume', 0.0)
                n_buyers_h.append(info.get('market/n_buyers', 0))
                if done or truncated: break
        
        res = {"seed": seed, "p2p": ep_p2p, "reward": ep_reward, "buyers": np.mean(n_buyers_h)}
        results.append(res)
        print(f"  Seed {seed:2d}: P2P={res['p2p']:>8.2f} kWh | Reward={res['reward']:>8.2f} | buyers={res['buyers']:.3f}")
        env.close()

    mean_p2p = np.mean([r['p2p'] for r in results])
    mean_buyers = np.mean([r['buyers'] for r in results])
    print(f"\n  Avg P2P: {mean_p2p:.2f} kWh | Avg Buyers: {mean_buyers:.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="research_q1/models/slim_v2/slim_v2_final")
    args = parser.parse_args()
    evaluate_v2(args.model, 4, 5)
