"""
End-to-end model validation script.
Uses synthetic_test_data.csv to verify the full training + evaluation pipeline.
"""
import sys
import numpy as np
sys.path.append('.')

from research_q1.novelty.slim_env import EnergyMarketEnvSLIM
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(data_file='synthetic_test_data.csv'):
    return EnergyMarketEnvSLIM(
        n_agents=4,
        data_file=data_file,
        enable_safety=True,
        enable_p2p=True,
        market_type='dynamic',
        alpha_p2p=0.005,
        base_delta=0.03,
        seed=0
    )


def run_validation():
    print("=" * 50)
    print("STEP 1: Training PPO for 10,000 timesteps on synthetic data")
    print("=" * 50)

    env = DummyVecEnv([make_env])
    model = PPO('MlpPolicy', env, n_steps=512, batch_size=64, verbose=0, seed=0)
    model.learn(total_timesteps=10_000)
    env.close()
    print("Training complete.")

    print("\n" + "=" * 50)
    print("STEP 2: Evaluating model for 200 steps")
    print("=" * 50)

    eval_env = DummyVecEnv([make_env])
    obs = eval_env.reset()
    p2p_total = 0.0
    grid_imp = 0.0
    grid_exp = 0.0
    rewards = []

    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, infos = eval_env.step(action)
        info = infos[0]
        p2p_total += info['p2p_volume']
        grid_imp  += info['grid_import']
        grid_exp  += info['grid_export']
        rewards.append(rew[0])

    final_profit = info['profit']
    eval_env.close()

    print(f"  P2P Volume (200 steps):   {p2p_total:.2f} kWh")
    print(f"  Grid Import (200 steps):  {grid_imp:.2f} kWh")
    print(f"  Grid Export (200 steps):  {grid_exp:.2f} kWh")
    print(f"  Avg Reward per step:      {np.mean(rewards):.4f}")
    print(f"  Min / Max Reward:         {np.min(rewards):.4f} / {np.max(rewards):.4f}")
    print(f"  Accumulated Profit:       ${final_profit:.2f}")

    print("\n" + "=" * 50)
    print("STEP 3: Sanity Checks")
    print("=" * 50)
    passed = True

    checks = [
        (p2p_total > 0,  "P2P trading occurred"),
        (grid_imp > 0,   "Grid import is non-zero"),
        (final_profit != 0.0, "Accumulated profit is non-zero"),
        (np.std(rewards) > 0, "Reward has variance (agent is not stuck)"),
    ]

    for condition, label in checks:
        status = "PASS" if condition else "FAIL"
        if not condition:
            passed = False
        print(f"  [{status}] {label}")

    print("\n" + ("ALL CHECKS PASSED — Pipeline is healthy." if passed
                  else "SOME CHECKS FAILED — Investigate above."))
    return passed


if __name__ == "__main__":
    run_validation()
