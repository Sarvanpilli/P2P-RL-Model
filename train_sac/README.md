# train_sac — SAC Experiment for P2P Energy Trading

This directory contains the complete **SAC (Soft Actor-Critic)** implementation
for the SLIM v2 P2P Energy Trading project, designed to be a drop-in
comparison against the existing PPO implementation.

---

## Files

| File | Purpose |
|:---|:---|
| `train_sac_hybrid.py` | Train a SAC agent with the same env & dataset as PPO |
| `evaluate_sac.py` | Evaluate a trained SAC model and save step-level metrics |
| `compare_ppo_vs_sac.py` | **Main comparison script** — runs both models, prints KPI table, saves plots |
| `results/` | Auto-created output directory for CSVs and plots |

---

## Quickstart

### Step 1 — Train SAC

```bash
# From the project root (same level as train/, train_sac/)
python train_sac/train_sac_hybrid.py --timesteps 300000 --seed 42
```

> Checkpoints saved to `models_sac/`.  
> TensorBoard logs saved to `tensorboard_logs/sac_hybrid/`.

Monitor live:
```bash
tensorboard --logdir=tensorboard_logs/
```

### Step 2 — Compare PPO vs SAC

```bash
python train_sac/compare_ppo_vs_sac.py \
    --ppo_model models_slim/seed_0/best_model.zip \
    --sac_model models_sac/best/best_model.zip \
    --n_episodes 5
```

> Outputs to `train_sac/results/`:
> - `comparison_summary.csv` — KPI table
> - `comparison_dashboard.png` — 4-panel plot
> - `plot_reward.png`, `plot_p2p_volume.png`, `plot_grid_import.png`
> - `bar_p2p_total.png`, `bar_profit_total.png`

### Step 3 — Evaluate SAC Only (optional)

```bash
python train_sac/evaluate_sac.py \
    --model_path models_sac/best/best_model.zip \
    --n_episodes 5
```

---

## Hyperparameter Comparison (PPO vs SAC)

| Parameter | PPO | SAC |
|:---|:---:|:---:|
| Type | On-policy | Off-policy |
| Buffer | Rollout (4096) | Replay (300K) |
| Batch size | 256 | 256 |
| Learning rate | 1e-4 | 3e-4 |
| Discount (γ) | 0.999 | 0.99 |
| Exploration | Entropy coef | Automatic α tuning |
| Network | [256, 256] LSTM | [400, 300] MLP |
| Dataset | processed_hybrid_data.csv | **Same** |
| Safety layer | AutonomousGuard + Lagrangian | **Same** |
