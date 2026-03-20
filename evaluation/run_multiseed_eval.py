"""
evaluation/run_multiseed_eval.py  (v3 — single checkpoint, 5 start-day seeds)

ALL models in research_q1/models/ were trained with EnergyMarketEnvSLIM.
Obs space: EnergyMarketEnvSLIM(forecast_horizon=2) → (88,).

Evaluation strategy (correct):
  - SLIM model:     slim_ppo_final.zip, 5 different start-day seeds
  - Baseline:       NoP2P ablation final_model.zip, 5 start-day seeds
  - Auction:        NoSafety ablation 100k checkpoint, 5 start-day seeds

Usage:
    python evaluation/run_multiseed_eval.py
"""

import os, sys, json, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

# ── Config ────────────────────────────────────────────────────────────────────
N_AGENTS    = 4
N_EVAL_STEPS = 1_752     # 20% test split = 8760 * 0.2
SEEDS       = [0, 1, 2, 3, 4]  # Used as random_start_day seeds
DATA_FILE   = "processed_hybrid_data.csv"

# Model paths — all research_q1 models, trained on EnergyMarketEnvSLIM
SLIM_PATH      = "research_q1/models/slim_ppo/slim_ppo_final.zip"
BASELINE_PATH  = "research_q1/models/slim_ablation_N4_NoP2P/final_model.zip"
AUCTION_PATH   = "research_q1/models/slim_ablation_N4_NoSafety/ppo_N4_NoSafety_100000_steps.zip"

OUTPUT_DIR = "evaluation/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_env(enable_safety, enable_p2p, seed, random_start_day=True):
    """Build the correct SLIM evaluation environment."""
    env = EnergyMarketEnvSLIM(
        n_agents=N_AGENTS,
        data_file=DATA_FILE,
        random_start_day=random_start_day,
        forecast_horizon=2,       # matches training config → obs=(88,)
        enable_safety=enable_safety,
        enable_p2p=enable_p2p,
        seed=seed,
    )
    return env


def run_one_seed(model, enable_safety, enable_p2p, seed):
    """Run N_EVAL_STEPS of deterministic evaluation for one start-day seed."""
    env = make_env(enable_safety, enable_p2p, seed, random_start_day=True)
    obs, _ = env.reset(seed=seed)

    p2p_total   = 0.0
    grid_total  = 0.0
    reward_sum  = 0.0
    safety_viol = 0

    for _ in range(N_EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        p2p_total   += info.get("p2p_volume", 0.0)
        grid_total  += info.get("grid_import", info.get("total_import", 0.0))
        reward_sum  += float(np.sum(reward))
        safety_viol += info.get("safety_violations", 0)

        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1000)

    env.close()
    return {
        "p2p_volume":        p2p_total,
        "grid_import":       grid_total,
        "avg_reward":        reward_sum / N_EVAL_STEPS,
        "safety_violations": safety_viol,
    }


def evaluate_model(name, model_path, enable_safety, enable_p2p):
    """Load model, run 5 seeds, return aggregated stats."""
    if not os.path.exists(model_path):
        print(f"  SKIP (not found): {model_path}")
        return None

    # Build a reference env for loading — just needs matching obs/act spaces
    ref_env = make_env(enable_safety, enable_p2p, seed=42, random_start_day=False)
    try:
        model = PPO.load(model_path, env=ref_env)
        ref_env.close()
        print(f"\nEvaluating {name}  [{os.path.basename(model_path)}]")
    except Exception as e:
        ref_env.close()
        print(f"  ERROR loading {name}: {e}")
        return None

    seed_results = []
    for seed in SEEDS:
        res = run_one_seed(model, enable_safety, enable_p2p, seed)
        seed_results.append(res)
        print(f"  seed={seed}: P2P={res['p2p_volume']:8.2f}  grid={res['grid_import']:8.2f}  "
              f"reward={res['avg_reward']:+.4f}  safety_violations={res['safety_violations']}")

    # Aggregate
    metrics = list(seed_results[0].keys())
    agg = {}
    for m in metrics:
        vals = [r[m] for r in seed_results]
        mean = np.mean(vals)
        std  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals))
        agg[m] = {"mean": float(mean), "std": float(std), "ci95": float(ci95), "n": len(vals)}

    return agg


def main():
    print("=" * 72)
    print("  Multi-Seed Evaluation  (single checkpoint × 5 start-day seeds)")
    print("=" * 72)

    all_results = {}

    # Evaluate all three in the correct env config
    for name, path, safety, p2p in [
        ("Baseline (Grid-only)", BASELINE_PATH, True,  False),  # NoP2P ablation
        ("Legacy Auction",       AUCTION_PATH,  False, True),   # NoSafety ablation
        ("SLIM (Full)",          SLIM_PATH,     True,  True),   # Full SLIM
    ]:
        agg = evaluate_model(name, path, safety, p2p)
        if agg:
            all_results[name] = agg

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    metrics = ["p2p_volume", "grid_import", "avg_reward", "safety_violations"]
    print(f"{'Metric':<22} | {'Baseline':>18} | {'Auction':>18} | {'SLIM':>18}")
    print("-" * 72)
    for m in metrics:
        row = []
        for group in ["Baseline (Grid-only)", "Legacy Auction", "SLIM (Full)"]:
            g = all_results.get(group, {}).get(m, {"mean": 0.0, "std": 0.0})
            row.append(f"{g['mean']:>8.2f} ± {g['std']:<6.2f}")
        print(f"{m:<22} | {row[0]:>18} | {row[1]:>18} | {row[2]:>18}")
    print("=" * 72)

    # ── Honest framing ────────────────────────────────────────────────────────
    slim_p2p      = all_results.get("SLIM (Full)", {}).get("p2p_volume", {}).get("mean", 0)
    auction_p2p   = all_results.get("Legacy Auction", {}).get("p2p_volume", {}).get("mean", 0)
    slim_p2p_std  = all_results.get("SLIM (Full)", {}).get("p2p_volume", {}).get("std", 0)
    auction_p2p_std = all_results.get("Legacy Auction", {}).get("p2p_volume", {}).get("std", 0)
    slim_grid     = all_results.get("SLIM (Full)", {}).get("grid_import", {}).get("mean", 0)
    base_grid     = all_results.get("Baseline (Grid-only)", {}).get("grid_import", {}).get("mean", 0)
    slim_grid_std = all_results.get("SLIM (Full)", {}).get("grid_import", {}).get("std", 0)

    print("\nHONEST FRAMING FOR DOCUMENTATION:")
    if slim_p2p > auction_p2p:
        pct = 100*(slim_p2p - auction_p2p)/max(auction_p2p, 0.01)
        print(f"  P2P: SLIM ({slim_p2p:.2f}±{slim_p2p_std:.2f} kWh) > Auction "
              f"({auction_p2p:.2f}±{auction_p2p_std:.2f} kWh) — {pct:.1f}% improvement ✓")
    else:
        print(f"  P2P: SLIM ({slim_p2p:.2f}±{slim_p2p_std:.2f} kWh) < Auction "
              f"({auction_p2p:.2f}±{auction_p2p_std:.2f} kWh) — SLIM trades less P2P")
    if slim_grid > base_grid:
        print(f"  Grid: SLIM ({slim_grid:.2f}±{slim_grid_std:.2f}) > Baseline ({base_grid:.2f}) "
              f"— SLIM is MORE grid-dependent at this training stage")
    else:
        print(f"  Grid: SLIM ({slim_grid:.2f}±{slim_grid_std:.2f}) < Baseline ({base_grid:.2f}) "
              f"— SLIM achieves grid reduction ✓")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results,
                   "n_eval_steps": N_EVAL_STEPS,
                   "n_seeds": len(SEEDS),
                   "seeds": SEEDS,
                   "timestamp": datetime.now().isoformat()}, f, indent=4)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
