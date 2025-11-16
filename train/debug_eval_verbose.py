# debug_eval_verbose.py
import os
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_env_improved import EnergyMarketEnv

# Get the parent directory (project root) and use models/ from there
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

MODEL_PATH = models_dir / "ppo_energy_continued.zip"
VECSTAT = models_dir / "vec_normalize.pkl"

# Try to load model, fallback to checkpoint if continued model doesn't exist
if not MODEL_PATH.exists():
    checkpoint_files = list(models_dir.glob("ppo_energy_checkpoint_*_steps.zip"))
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        MODEL_PATH = checkpoint_files[0]
        print(f"Continued model not found, using checkpoint: {MODEL_PATH.name}")
    else:
        raise FileNotFoundError(f"No model files found in {models_dir}")

# Load model first to inspect its observation space
print("Inspecting saved model to determine environment configuration...")
try:
    temp_model = PPO.load(str(MODEL_PATH), print_system_info=False)
    saved_obs_space = temp_model.observation_space
    obs_size = saved_obs_space.shape[0]
    print(f"Saved model observation space: {saved_obs_space.shape}")
except Exception as e:
    print(f"Warning: Could not inspect model ({e}), assuming 20 dimensions (forecast_horizon=0)")
    obs_size = 20

# Calculate what forecast_horizon was used based on observation space
# obs_size = n_agents * (5 + 2*forecast_horizon)
# So: forecast_horizon = (obs_size / n_agents - 5) / 2
N_AGENTS = 4
forecast_horizon = int((obs_size / N_AGENTS - 5) / 2)
if forecast_horizon < 0:
    forecast_horizon = 0
print(f"Inferred forecast_horizon: {forecast_horizon} (for {N_AGENTS} agents, obs_size={obs_size})")

def make_vec(forecast_horizon=0):
    return DummyVecEnv([lambda: EnergyMarketEnv(n_agents=N_AGENTS, 
                                                max_line_capacity_kw=200.0,
                                                per_agent_max_kw=120.0, 
                                                base_price=0.12,
                                                price_slope=0.002, 
                                                overload_multiplier=25.0,
                                                forecast_horizon=forecast_horizon)])

vec = make_vec(forecast_horizon=forecast_horizon)

# Try to load VecNormalize stats with error handling
if VECSTAT.exists():
    try:
        vec = VecNormalize.load(str(VECSTAT), vec)
        vec.training = False
        vec.norm_reward = False
        print("Successfully loaded VecNormalize stats.")
    except (AssertionError, ValueError) as e:
        print(f"Warning: Could not load VecNormalize stats (shape mismatch: {e})")
        print("Running without normalization.")
else:
    print("VecNormalize not found, running unnormalized.")

# Load model with matching environment
model = PPO.load(str(MODEL_PATH), env=vec)

# VecEnv.reset() returns only observation, not (obs, info) tuple
obs = vec.reset()
action, _ = None, None

# run a single episode and print verbose info
for t in range(120):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec.step(action)
    # VecEnv returns dones (not terminated/truncated separately)
    terminated = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
    truncated = False  # VecEnv doesn't distinguish terminated/truncated
    info = infos[0] if isinstance(infos, list) else infos
    # rewards may be array or scalar
    r = float(np.array(rewards).reshape(-1)[0]) if hasattr(rewards, "__len__") else float(rewards)
    # info may be dict or list
    ginfo = info if isinstance(info, dict) else info[0]
    # extract per-agent diagnostics if present
    intended = ginfo.get("intended_injection_kw")
    curtail = ginfo.get("curtailment_kw")
    soc = None
    pv = None
    # try to extract soc and pv from observation (obs flattened)
    flat = np.array(obs).reshape(-1)
    # obs layout per agent: [demand,soc,pv, total_export,total_import, pv_f1.., dem_f1..]
    per_len = (len(flat) // 4)  # attempt to infer per-agent length (fallback)
    n_agents = 4
    per_len = len(flat) // n_agents
    per = flat.reshape(n_agents, per_len)
    try:
        soc = per[:,1]
        pv = per[:,2]
    except Exception:
        soc = np.zeros(n_agents)
        pv = np.zeros(n_agents)

    print(f"\nStep {t}  global_price={ginfo.get('market_price'):.4f} line_ol={ginfo.get('line_overload_kw'):.4f} total_reward={r:.3f}")
    for i in range(n_agents):
        a_batt = action.reshape(n_agents,2)[i,0]
        a_grid = action.reshape(n_agents,2)[i,1]
        intended_i = float(intended[i]) if intended is not None else None
        curtail_i = float(curtail[i]) if curtail is not None else 0.0
        print(f" A{i}: action=[batt {a_batt:.2f}, grid {a_grid:.2f}] soc={soc[i]:.2f} pv={pv[i]:.2f} intended={intended_i} curtail={curtail_i}")
    if terminated or truncated:
        print("Episode ended")
        break
