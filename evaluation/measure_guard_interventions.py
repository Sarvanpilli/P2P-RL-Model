"""
evaluation/measure_guard_interventions.py

Measures how often AutonomousGuard (layer2 / FeasibilityFilter) modifies
the raw PPO action at four training checkpoints.

Usage:
    python evaluation/measure_guard_interventions.py

Outputs:
    - Prints a result table to stdout
    - Saves evaluation/results/guard_intervention_rates.csv
"""

import os
import sys
import csv
import numpy as np
from stable_baselines3 import PPO

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.energy_env_robust import EnergyMarketEnvRobust
from train.autonomous_guard import AutonomousGuard

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "models_phase5_hybrid"
CHECKPOINTS = [50_000, 100_000, 150_000, 200_000]
EVAL_STEPS = 1_000
OUTPUT_DIR = "evaluation/results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "guard_intervention_rates.csv")


def make_env():
    """Return a single (non-vectorised) evaluation environment."""
    env = EnergyMarketEnvRobust()
    return env


def measure_checkpoint(checkpoint_path: str, env, guard: AutonomousGuard,
                        n_steps: int = EVAL_STEPS):
    """Load one checkpoint and count guard interventions over n_steps."""
    model = PPO.load(checkpoint_path, env=env)

    obs, _ = env.reset()
    interventions = 0
    total = 0

    for _ in range(n_steps):
        # Raw action from policy (deterministic greedy)
        raw_action, _ = model.predict(obs, deterministic=True)

        # Get current state for guard
        state = env.get_state() if hasattr(env, "get_state") else None

        # Apply guard — uses FeasibilityFilter internally
        safe_action, guard_info = guard.guard_action(raw_action, state)

        # Detect any change introduced by the guard
        changed = not np.allclose(
            raw_action.flatten(), safe_action.flatten(), atol=1e-5
        )
        if changed:
            interventions += 1

        obs, _, terminated, truncated, _ = env.step(safe_action)
        total += 1

        if terminated or truncated:
            obs, _ = env.reset()

    return interventions, total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_env()
    guard = AutonomousGuard(n_agents=env.n_agents if hasattr(env, "n_agents") else 4)

    rows = []
    header = ["Checkpoint", "Steps", "Interventions", "Total Steps", "Rate (%)"]
    print(f"\n{'Checkpoint':<12} {'Steps':>8} {'Interventions':>14} "
          f"{'Total Steps':>12} {'Rate (%)':>10}")
    print("-" * 62)

    for steps in CHECKPOINTS:
        ckpt_name = f"ppo_hybrid_{steps}_steps.zip"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"  [{steps//1000}k] SKIPPED — file not found: {ckpt_path}")
            continue

        label = f"{steps // 1000}k"
        interventions, total = measure_checkpoint(ckpt_path, env, guard)
        rate = 100.0 * interventions / total if total > 0 else 0.0

        row = [label, steps, interventions, total, f"{rate:.1f}"]
        rows.append(row)
        print(f"  {label:<10} {steps:>8} {interventions:>14} {total:>12} {rate:>9.1f}%")

    print()
    env.close()

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
