"""
evaluate_sac.py
================
Evaluate a trained SAC model on the P2P Energy Trading environment and
collect step-level metrics to a CSV for comparison with PPO.

Usage:
    python train_sac/evaluate_sac.py --model_path models_sac/sac_hybrid_final.zip
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import SAC
from train.energy_env_robust import EnergyMarketEnvRobust


# ─────────────────────────────────────────────────────────────────────────────
def make_eval_env(data_file: str, seed: int = 0) -> EnergyMarketEnvRobust:
    """Non-randomised env for reproducible evaluation."""
    env = EnergyMarketEnvRobust(
        n_agents=4,
        data_file=data_file,
        random_start_day=False,
        enable_ramp_rates=True,
        enable_losses=True,
        forecast_horizon=4,
        enable_predictive_obs=True,
        forecast_noise_std=0.0,
        diversity_mode=True,
    )
    return env


def evaluate_model(
    model_path: str,
    data_file: str = "processed_hybrid_data.csv",
    n_episodes: int = 5,
    output_csv: str = "train_sac/results/sac_eval_results.csv",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run the SAC model for n_episodes and return a DataFrame with per-step metrics.
    """
    print(f"\n[EVAL-SAC] Loading model from: {model_path}")
    model = SAC.load(model_path)

    all_rows = []

    for ep in range(n_episodes):
        env = make_eval_env(data_file, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step = 0
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            row = {
                "episode": ep,
                "step": step,
                "algorithm": "SAC",
                "reward": float(reward),
                "p2p_volume_kwh": info.get("p2p_volume_kwh_step", 0.0),
                "total_export_kw": info.get("total_export", 0.0),
                "total_import_kw": info.get("total_import", 0.0),
                "market_price": info.get("market_price", 0.0),
                "line_overload_kw": info.get("line_overload_kw", 0.0),
                "failed_trades": info.get("failed_trades", 0),
                "lagrangian_soc_violation": info.get("lagrangian/violation_soc", 0.0),
                "lagrangian_line_violation": info.get("lagrangian/violation_line", 0.0),
                "absolute_profit_usd": info.get("absolute_profit_usd", 0.0),
            }
            all_rows.append(row)
            ep_reward += float(reward)
            step += 1

        print(f"  Episode {ep+1}/{n_episodes} | Steps: {step} | Total Reward: {ep_reward:.2f}")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[EVAL-SAC] Results saved → {output_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC model")
    parser.add_argument("--model_path", type=str, default="models_sac/sac_hybrid_final.zip",
                        help="Path to the SAC .zip checkpoint")
    parser.add_argument("--data_file", type=str, default="processed_hybrid_data.csv",
                        help="Dataset CSV")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--output_csv", type=str,
                        default="train_sac/results/sac_eval_results.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = evaluate_model(
        model_path=args.model_path,
        data_file=args.data_file,
        n_episodes=args.n_episodes,
        output_csv=args.output_csv,
        seed=args.seed,
    )

    print("\n[EVAL-SAC] Summary:")
    print(df.groupby("algorithm")[["reward", "p2p_volume_kwh", "absolute_profit_usd"]].mean().round(4))
