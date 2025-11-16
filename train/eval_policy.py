# eval_policy.py
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

# Try to load final model, fallback to latest checkpoint
model_path = models_dir / "ppo_energy_final.zip"
if not model_path.exists():
    # Use the latest checkpoint if final model doesn't exist
    checkpoint_files = list(models_dir.glob("ppo_energy_checkpoint_*_steps.zip"))
    if checkpoint_files:
        # Sort by modification time and use the latest
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = checkpoint_files[0]
        print(f"Final model not found, using checkpoint: {model_path.name}")
    else:
        raise FileNotFoundError(f"No model files found in {models_dir}")

# load model
model = PPO.load(str(model_path))

# load VecNormalize stats if used
vecnorm_path = models_dir / "vec_normalize.pkl"
try:
    venv = DummyVecEnv([lambda: EnergyMarketEnv(n_agents=4, max_line_capacity_kw=200.0, 
                                                per_agent_max_kw=120.0, base_price=0.12,
                                                price_slope=0.002, overload_multiplier=25.0)])
    if vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("VecNormalize stats not found, using unnormalized environment")
except Exception as e:
    print("VecNormalize load failed (fallback):", e)
    venv = DummyVecEnv([lambda: EnergyMarketEnv(n_agents=4, max_line_capacity_kw=200.0,
                                                per_agent_max_kw=120.0, base_price=0.12,
                                                price_slope=0.002, overload_multiplier=25.0)])

# run deterministic episodes
for ep in range(5):
    obs = venv.reset()  # VecEnv returns only observation, not tuple
    done = False
    total_reward = 0.0
    step = 0
    while step < 200 and not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)  # VecEnv returns dones (not terminated/truncated separately)
        # VecEnv always returns arrays
        reward_value = float(rewards[0])
        total_reward += reward_value
        done = bool(dones[0])
        step += 1
        if done:
            break
    print(f"Episode {ep} reward: {total_reward}")
