# resume_parallel.py
import os
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from energy_env_improved import EnergyMarketEnv

def make_env(seed_offset=0, n_agents=4, seed=123, forecast_horizon=0):
    def _init():
        env = EnergyMarketEnv(n_agents=n_agents,
                              max_line_capacity_kw=200.0,
                              per_agent_max_kw=120.0,
                              base_price=0.12,
                              price_slope=0.002,  # Match original training parameters
                              overload_multiplier=25.0,  # Match original training parameters
                              forecast_horizon=forecast_horizon,
                              seed=seed + seed_offset)
        return Monitor(env)
    return _init

if __name__ == '__main__':
    # Get the parent directory (project root) and use models/ from there
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    MODEL_DIR = os.path.abspath(MODEL_DIR)  # Normalize the path
    os.makedirs(MODEL_DIR, exist_ok=True)

    # find latest checkpoint (look for ppo_energy_checkpoint_*.zip or ppo_energy_continued.zip)
    candidates = glob.glob(os.path.join(MODEL_DIR, "ppo_energy_checkpoint_*.zip")) + \
                 glob.glob(os.path.join(MODEL_DIR, "ppo_energy_continued*.zip")) + \
                 glob.glob(os.path.join(MODEL_DIR, "ppo_energy_*.zip"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found in models/. Run initial training first.")
    # pick the most recently modified
    checkpoint = max(candidates, key=os.path.getmtime)
    print("Resuming from checkpoint:", checkpoint)

    # Load model first without environment to inspect its observation space
    print("Inspecting saved model to determine environment configuration...")
    try:
        # Try loading without env to get observation space info
        import zipfile
        import pickle
        with zipfile.ZipFile(checkpoint, 'r') as archive:
            # Load the data.pkl which contains model metadata
            with archive.open('data.pkl', 'r') as f:
                data = pickle.load(f)
                saved_obs_space = data.get('observation_space')
                if saved_obs_space is not None:
                    obs_size = saved_obs_space.shape[0]
                    print(f"Saved model observation space: {saved_obs_space.shape}")
                else:
                    raise ValueError("Could not find observation_space in saved model")
    except Exception as e:
        print(f"Warning: Could not inspect model file directly ({e}), trying alternative method...")
        # Fallback: try loading model without env (may fail but we'll catch it)
        try:
            temp_model = PPO.load(checkpoint, print_system_info=False)
            saved_obs_space = temp_model.observation_space
            obs_size = saved_obs_space.shape[0]
            print(f"Saved model observation space: {saved_obs_space.shape}")
        except Exception as e2:
            # Last resort: assume default based on error message pattern
            print(f"Could not determine observation space, assuming 20 dimensions (forecast_horizon=0)")
            obs_size = 20
    
    # Calculate what forecast_horizon was used based on observation space
    # obs_size = n_agents * (5 + 2*forecast_horizon)
    # So: forecast_horizon = (obs_size / n_agents - 5) / 2
    N_AGENTS = 4  # Assume 4 agents (can be adjusted if needed)
    forecast_horizon = int((obs_size / N_AGENTS - 5) / 2)
    if forecast_horizon < 0:
        forecast_horizon = 0
    print(f"Inferred forecast_horizon: {forecast_horizon} (for {N_AGENTS} agents, obs_size={obs_size})")

    VECSTAT = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    N_PARALLEL = 4
    SEED = 123
    CONTINUE_TIMESTEPS = 1_000_000  # change if you want more or less

    print(f"Creating SubprocVecEnv with {N_PARALLEL} processes...")
    env_fns = [make_env(i, n_agents=N_AGENTS, seed=SEED, forecast_horizon=forecast_horizon) 
               for i in range(N_PARALLEL)]
    vec_env = SubprocVecEnv(env_fns)

    if os.path.exists(VECSTAT):
        print("Attempting to load VecNormalize stats...")
        try:
            vec_env = VecNormalize.load(VECSTAT, vec_env)
            print("Successfully loaded VecNormalize stats.")
        except (AssertionError, ValueError) as e:
            print(f"Warning: Could not load VecNormalize stats (shape mismatch or other error: {e})")
            print("Creating new VecNormalize wrapper...")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        print("No VecNormalize stats found. Creating new VecNormalize wrapper...")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Loading PPO model with matching environment...")
    model = PPO.load(checkpoint, env=vec_env)
    # ensure model uses new env
    model.set_env(vec_env)
    # Note: Don't change n_steps after loading - the buffer was created with the original n_steps
    # If you need to change n_steps, you'd need to recreate the model with the new value
    print(f"Using n_steps={model.n_steps} from saved model")

    print("Continuing training for", CONTINUE_TIMESTEPS, "timesteps...")
    # Use reset_num_timesteps=True to properly initialize the rollout buffer
    # The learned weights are preserved, only the timestep counter resets
    model.learn(total_timesteps=CONTINUE_TIMESTEPS, reset_num_timesteps=True)

    final_path = os.path.join(MODEL_DIR, "ppo_energy_continued_parallel.zip")
    print("Saving model to", final_path)
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print("Done.")
