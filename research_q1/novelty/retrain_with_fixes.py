"""
retrain_with_fixes.py — Retrain SLIM with Nash-Equilibrium Fixes
================================================================

Implements three targeted fixes for the all-seller policy collapse:
  Fix 1: P2P completion bonus (both buyer AND seller get rewarded)
  Fix 2: Role-diversity penalty (no-buyer / no-seller market is penalised)
  Fix 3: market_balance obs feature (agents can see if market needs buyers)
         → obs_dim 104 → 105

Curriculum:
  Stage 1  (0 – 50 k steps): aggressive P2P incentives, no Lagrangian
  Stage 2  (50k – 150k steps): moderate incentives, Lagrangian ON (α=0.001)
  Stage 3  (150k – 300k steps): realistic incentives, Lagrangian ON (α=0.005)

TensorBoard keys logged every step:
  reward/p2p_bonus_mean        — avg P2P completion bonus across steps
  reward/no_buyer_penalty_mean — avg role-diversity penalty
  market/n_buyers_mean         — avg number of buyers per step
  market/n_sellers_mean        — avg number of sellers per step
  market/p2p_volume_mean       — avg P2P kWh per step

Usage:
  cd f:/Projects/P2P-RL-Model
  python research_q1/novelty/retrain_with_fixes.py

Fix is working when:
  market/n_buyers_mean > 0.5 after 50 k steps
If still 0 → increase Stage 1 p2p_bonus to 0.50 and re-run.
"""

import os
import sys
import numpy as np
import torch
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)

# ─── Path fix ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM


# ─── Hyper-parameters ────────────────────────────────────────────────────────
N_AGENTS        = 4
TOTAL_TIMESTEPS = 300_000
SEED            = 42
MODEL_DIR       = "research_q1/models/slim_v2"
LOG_DIR         = "research_q1/logs/slim_v2_tensorboard"

# Curriculum stage boundaries
STAGE_1_END = 50_000
STAGE_2_END = 150_000

# Stage reward weights  (p2p_bonus, no_buyer_penalty, grid_penalty)
STAGE_1_WEIGHTS = dict(p2p_bonus=0.30, no_buyer_penalty=0.20, grid_penalty=0.05)
STAGE_2_WEIGHTS = dict(p2p_bonus=0.20, no_buyer_penalty=0.15, grid_penalty=0.15)
STAGE_3_WEIGHTS = dict(p2p_bonus=0.15, no_buyer_penalty=0.10, grid_penalty=0.20)

# Lagrangian multiplier update rate (0 = OFF in Stage 1)
LAGRANGIAN_ALPHA_S2 = 0.001
LAGRANGIAN_ALPHA_S3 = 0.005


# ─── Curriculum Callback ─────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Adjusts reward weights at curriculum stage boundaries and logs extra
    market-health metrics to TensorBoard at every logging interval.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._current_stage = 0

        # Rolling windows for smooth TensorBoard curves
        self._window = 500  # steps
        self._p2p_bonus_buf      = deque(maxlen=self._window)
        self._no_buyer_pen_buf   = deque(maxlen=self._window)
        self._n_buyers_buf       = deque(maxlen=self._window)
        self._n_sellers_buf      = deque(maxlen=self._window)
        self._p2p_volume_buf     = deque(maxlen=self._window)

    def _on_training_start(self) -> None:
        # Apply stage-1 weights immediately
        self._apply_stage(1)

    def _apply_stage(self, stage: int) -> None:
        if stage == self._current_stage:
            return
        self._current_stage = stage
        weights = {1: STAGE_1_WEIGHTS, 2: STAGE_2_WEIGHTS, 3: STAGE_3_WEIGHTS}[stage]
        self.training_env.env_method('set_reward_weights', **weights)
        if self.verbose:
            print(f"\n[Curriculum] Stage {stage} activated at step "
                  f"{self.num_timesteps:,}  → weights={weights}")

    def _on_step(self) -> bool:
        n = self.num_timesteps

        # ── Stage transitions ──────────────────────────────────────────────
        if n < STAGE_1_END:
            self._apply_stage(1)
        elif n < STAGE_2_END:
            self._apply_stage(2)
        else:
            self._apply_stage(3)

        # ── Collect metrics from info dict ────────────────────────────────
        # self.locals['infos'] is a list of dicts (one per vec-env)
        for info in self.locals.get('infos', []):
            self._p2p_bonus_buf.append(info.get('reward/p2p_bonus', 0.0))
            self._no_buyer_pen_buf.append(info.get('reward/no_buyer_penalty', 0.0))
            self._n_buyers_buf.append(info.get('market/n_buyers', 0))
            self._n_sellers_buf.append(info.get('market/n_sellers', 0))
            self._p2p_volume_buf.append(info.get('market/p2p_volume', 0.0))

        # ── Log to TensorBoard every 500 steps ───────────────────────────
        if n % 500 == 0 and len(self._n_buyers_buf) > 0:
            self.logger.record(
                "reward/p2p_bonus_mean",
                float(np.mean(self._p2p_bonus_buf)))
            self.logger.record(
                "reward/no_buyer_penalty_mean",
                float(np.mean(self._no_buyer_pen_buf)))
            self.logger.record(
                "market/n_buyers_mean",
                float(np.mean(self._n_buyers_buf)))
            self.logger.record(
                "market/n_sellers_mean",
                float(np.mean(self._n_sellers_buf)))
            self.logger.record(
                "market/p2p_volume_mean",
                float(np.mean(self._p2p_volume_buf)))

            # Quick-health check printed to console
            if self.verbose >= 1 and n % 10_000 == 0:
                n_buy = np.mean(self._n_buyers_buf)
                p2p_v = np.mean(self._p2p_volume_buf)
                stage = self._current_stage
                ok = "✓ FIX WORKING" if n_buy > 0.5 else "✗ still no buyers"
                print(f"  [step {n:>8,}] Stage={stage} | "
                      f"buyers={n_buy:.2f} | p2p_vol={p2p_v:.3f} kWh | {ok}")

        return True


# ─── Environment factory ──────────────────────────────────────────────────────
def make_env(seed: int = SEED):
    def _init():
        env = EnergyMarketEnvSLIM(
            n_agents=N_AGENTS,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True,
            seed=seed,
        )
        # Immediately apply Stage-1 weights so the very first episodes use them
        env.set_reward_weights(**STAGE_1_WEIGHTS)
        return env
    return _init


# ─── Main ─────────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("=" * 60)
    print("  SLIM v2  —  Nash-Equilibrium Fix Retraining")
    print("=" * 60)
    print(f"  N_agents        : {N_AGENTS}")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Model output    : {MODEL_DIR}")
    print(f"  TensorBoard     : {LOG_DIR}")
    print()

    # Build vec-env
    env = DummyVecEnv([make_env(SEED)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Sanity-check obs_dim
    obs_dim = env.observation_space.shape[0]
    print(f"  Observation space: ({obs_dim},)  [expected 105 for N=4, H=4]")
    if obs_dim != 105:
        print(f"  ⚠  obs_dim={obs_dim} — if N or H differs from defaults, this is expected.")

    # ── Callbacks ─────────────────────────────────────────────────────────
    curriculum_cb = CurriculumCallback(verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="slim_v2",
        verbose=1,
    )

    # ── PPO model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,          # increased from 64 for stability
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,           # increased from 0.01 — more exploration to escape all-seller trap
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        device="auto",
    )

    print(f"\n  Policy: {model.policy.__class__.__name__}")
    print(f"  ent_coef=0.02 (↑ from 0.01) — extra exploration to escape sell-trap")
    print(f"  batch_size=256 (↑ from 64)  — more stable gradients\n")

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[curriculum_cb, checkpoint_cb],
            progress_bar=True,
            tb_log_name="slim_v2_run",
        )
        print("\n[✓] Training complete.")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    finally:
        # Save final model and normalisation stats
        final_model_path = os.path.join(MODEL_DIR, "slim_v2_final")
        norm_path        = os.path.join(MODEL_DIR, "vec_normalize_v2.pkl")

        model.save(final_model_path)
        env.save(norm_path)

        print(f"[✓] Model saved  → {final_model_path}.zip")
        print(f"[✓] VecNormalize → {norm_path}")
        print()
        print("Next steps:")
        print("  1. Check TensorBoard: tensorboard --logdir", LOG_DIR)
        print("  2. Run evaluation:    python research_q1/eval/evaluate_v2.py")
        print("  3. Key metric:        market/n_buyers_mean > 0.5  = fix is working")


if __name__ == "__main__":
    train()
