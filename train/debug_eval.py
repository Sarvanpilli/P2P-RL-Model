# debug_eval.py
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# import your env - handle both running from root and from train directory
try:
    from train.energy_env_improved import EnergyMarketEnv
except ImportError:
    # If running from train directory, use relative import
    from energy_env_improved import EnergyMarketEnv

# Get the project root directory (parent of train directory)
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

MODEL_PATH = models_dir / "ppo_energy_continued.zip"   # update if different
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

# create vec env with matching forecast_horizon
def make_env(forecast_horizon=0):
    return DummyVecEnv([lambda: EnergyMarketEnv(n_agents=N_AGENTS, max_line_capacity_kw=200.0,
                                                per_agent_max_kw=120.0, base_price=0.12,
                                                price_slope=0.002, overload_multiplier=25.0,
                                                forecast_horizon=forecast_horizon)])

vec = make_env(forecast_horizon=forecast_horizon)

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
    print("VecNormalize stats not found - running unnormalized.")

# load model with matching environment
model = PPO.load(str(MODEL_PATH), env=vec)

EPISODES = 5
MAX_STEPS = 200

for ep in range(EPISODES):
    obs = vec.reset()  # VecEnv returns only observation, not tuple
    total_reward = 0.0
    for t in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec.step(action)  # VecEnv returns dones, not terminated/truncated
        total_reward += float(rewards[0])
        done = bool(dones[0])
        # print per-step diagnostics (info is same for all agents in this env)
        if t % 10 == 0 or done:
            global_info = infos[0] if isinstance(infos, list) else infos
            print(f"ep{ep} step{t} | price={global_info.get('market_price', 0):.4f} | "
                  f"line_overload={global_info.get('line_overload_kw', 0):.4f} | "
                  f"export_before={global_info.get('total_export_kw_before', 0):.2f} "
                  f"import_before={global_info.get('total_import_kw_before', 0):.2f} | reward={float(rewards[0]):.3f}")
        if done:
            break
    print(f"Episode {ep} total reward: {total_reward:.4f}\n")
