"""
debug_grid_import.py

Runs SLIM (slim_ppo_final) for 168 hours and diagnoses high grid import.
Prints:
  - per-step battery actions, trade actions, grid flows, p2p volume, hour
  - summary: avg battery action per agent type
  - hours with highest grid import
  - whether P2P trades occur at all
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.abspath("."))

from research_q1.novelty.slim_env import EnergyMarketEnvSLIM
from stable_baselines3 import PPO

CHECKPOINT = "research_q1/models/slim_ppo/slim_ppo_final.zip"
N_STEPS = 168  # 1 week
N_AGENTS = 4
AGENT_TYPES = ["Solar", "Wind", "EV/V2G", "Standard"]

# Build env that matches training config
env = EnergyMarketEnvSLIM(
    n_agents=N_AGENTS,
    data_file="processed_hybrid_data.csv",
    random_start_day=False,
    forecast_horizon=2,
    enable_safety=True,
    enable_p2p=True,
)

try:
    model = PPO.load(CHECKPOINT, env=env)
    print(f"Model loaded: {CHECKPOINT}")
    print(f"  obs={model.observation_space.shape}  act={model.action_space.shape}\n")
except Exception as e:
    print(f"LOAD ERROR: {e}")
    sys.exit(1)

obs, _ = env.reset(seed=0)

# Per-step storage
records = []

print(f"{'hr':>3} | {'bat[0]':>7} {'bat[1]':>7} {'bat[2]':>7} {'bat[3]':>7} "
      f"| {'trd[0]':>7} {'trd[1]':>7} {'trd[2]':>7} {'trd[3]':>7} "
      f"| {'p2p':>7} {'g_imp':>7} {'g_exp':>7}")
print("-"*100)

for t in range(N_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    hour = t % 24
    p2p  = info.get("p2p_volume", 0.0)
    g_imp = info.get("grid_import", info.get("total_import", 0.0))
    g_exp = info.get("grid_export", info.get("total_export", 0.0))

    raw = action.reshape(N_AGENTS, 2)
    bat = raw[:, 0]
    trd = raw[:, 1]

    records.append({
        "t": t, "hour": hour,
        "bat": bat.copy(), "trd": trd.copy(),
        "p2p": p2p, "grid_import": g_imp, "grid_export": g_exp,
        "reward": float(np.sum(reward)),
    })

    if t < 48:  # Print first 2 days
        bat_str = " ".join(f"{b:+.3f}" for b in bat)
        trd_str = " ".join(f"{tr:+.3f}" for tr in trd)
        print(f"{hour:3d} | {bat_str} | {trd_str} | {p2p:7.3f} {g_imp:7.3f} {g_exp:7.3f}")

    if terminated or truncated:
        obs, _ = env.reset(seed=t)

env.close()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

bats = np.array([r["bat"] for r in records])   # (168, 4)
trds = np.array([r["trd"] for r in records])   # (168, 4)
p2p_arr = np.array([r["p2p"] for r in records])
g_imp_arr = np.array([r["grid_import"] for r in records])

print("\n--- Average Battery Action per Agent Type ---")
for i, name in enumerate(AGENT_TYPES):
    avg_b = bats[:, i].mean()
    pct_charge = (bats[:, i] > 0.05).mean() * 100
    pct_discharge = (bats[:, i] < -0.05).mean() * 100
    pct_idle = 100 - pct_charge - pct_discharge
    print(f"  Agent {i} ({name:10s}): mean={avg_b:+.3f}  "
          f"charge={pct_charge:.0f}%  discharge={pct_discharge:.0f}%  idle={pct_idle:.0f}%")

print("\n--- Average Trade Action per Agent Type ---")
for i, name in enumerate(AGENT_TYPES):
    avg_t = trds[:, i].mean()
    pct_sell = (trds[:, i] > 0.05).mean() * 100
    pct_buy  = (trds[:, i] < -0.05).mean() * 100
    print(f"  Agent {i} ({name:10s}): trade_mean={avg_t:+.3f}  "
          f"selling={pct_sell:.0f}%  buying={pct_buy:.0f}%")

print("\n--- Grid Import by Hour of Day ---")
by_hour = {h: [] for h in range(24)}
for r in records:
    by_hour[r["hour"]].append(r["grid_import"])
peak_hours = sorted(range(24), key=lambda h: np.mean(by_hour[h]), reverse=True)[:6]
print(f"  Top 6 hours with highest grid import: {peak_hours}")
for h in peak_hours:
    vals = by_hour[h]
    print(f"    Hour {h:02d}: mean={np.mean(vals):.3f}  max={np.max(vals):.3f}  n={len(vals)}")

print("\n--- P2P Trading Summary ---")
p2p_steps = (p2p_arr > 0.01).sum()
print(f"  Steps with P2P > 0.01:  {p2p_steps}/{N_STEPS} ({100*p2p_steps/N_STEPS:.1f}%)")
print(f"  Total P2P volume:        {p2p_arr.sum():.3f} kWh")
print(f"  Total Grid Import:       {g_imp_arr.sum():.3f} kWh")
if g_imp_arr.sum() > 0:
    ratio = p2p_arr.sum() / g_imp_arr.sum()
    print(f"  P2P / Grid ratio:        {ratio:.4f} ({ratio*100:.2f}%)")

print("\n--- Diagnosis ---")
if p2p_steps < 5:
    print("  FINDING: Agents are NOT trading P2P. They are relying entirely on the grid.")
    avg_sell = (trds > 0.05).mean()
    avg_buy = (trds < -0.05).mean()
    if avg_sell < 0.05 and avg_buy < 0.05:
        print("  ROOT CAUSE: Trade actions are near zero — agent learned to NOT trade.")
    else:
        print("  ROOT CAUSE: Trade actions present but market clearing not producing volume.")
        print("  Check: safety layer is blocking trades, or market demand < supply always.")
elif g_imp_arr.sum() > 300:
    print("  FINDING: P2P exists but grid import is still very high.")
    print("  ROOT CAUSE: Grid is being used as primary energy source rather than P2P/battery.")
    if bats.mean() > 0:
        print("  Battery actions: mostly charging (agents storing rather than supplying).")
    else:
        print("  Battery actions: mostly discharging — agents using own battery, still need grid.")
