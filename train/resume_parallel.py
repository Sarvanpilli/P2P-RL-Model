# resume_parallel.py
import os
import glob
import platform
import multiprocessing as mp
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from energy_env_improved import EnergyMarketEnv

def make_env(seed_offset=0, n_agents=4, seed=123, forecast_horizon=0, shaping_coef=1.0,
             battery_capacity_kwh=150.0, battery_max_charge_kw=80.0):
    def _init():
        env = EnergyMarketEnv(n_agents=n_agents,
                              max_line_capacity_kw=200.0,
                              per_agent_max_kw=120.0,
                              base_price=0.12,
                              price_slope=0.002,
                              overload_multiplier=25.0,
                              forecast_horizon=forecast_horizon,
                              seed=seed + seed_offset,
                              shaping_coef=shaping_coef,
                              battery_capacity_kwh=battery_capacity_kwh,
                              battery_max_charge_kw=battery_max_charge_kw)
        return Monitor(env)
    return _init

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    candidates = glob.glob(os.path.join(MODEL_DIR, "ppo_energy_checkpoint_*.zip")) + \
                 glob.glob(os.path.join(MODEL_DIR, "ppo_energy_continued*.zip")) + \
                 glob.glob(os.path.join(MODEL_DIR, "ppo_energy_*.zip"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found in models/. Run initial training first.")
    checkpoint = max(candidates, key=os.path.getmtime)
    print("Resuming from checkpoint:", checkpoint)

    print("Inspecting saved model to determine environment configuration...")
    obs_size = None
    try:
        import zipfile, pickle
        with zipfile.ZipFile(checkpoint, 'r') as archive:
            with archive.open('data.pkl', 'r') as f:
                data = pickle.load(f)
                saved_obs_space = data.get('observation_space')
                if saved_obs_space is not None:
                    obs_size = saved_obs_space.shape[0]
                    print(f"Saved model observation space: {saved_obs_space.shape}")
    except Exception as e:
        print(f"Warning: quick inspect failed ({e}), falling back to loading model metadata...")
        try:
            temp_model = PPO.load(checkpoint, print_system_info=False)
            saved_obs_space = getattr(temp_model, "observation_space", None)
            if saved_obs_space is not None:
                obs_size = saved_obs_space.shape[0]
                print(f"Saved model observation space via load: {saved_obs_space.shape}")
        except Exception as e2:
            print(f"Warning: fallback load failed ({e2}), assuming obs_size=20.")
            obs_size = 20

    N_AGENTS = 4
    forecast_horizon = int((obs_size / N_AGENTS - 5) / 2)
    if forecast_horizon < 0:
        forecast_horizon = 0
    print(f"Inferred forecast_horizon: {forecast_horizon} (for {N_AGENTS} agents, obs_size={obs_size})")

    VECSTAT = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    N_PARALLEL = int(os.environ.get("N_PARALLEL", "4"))
    SEED = int(os.environ.get("SEED", "123"))
    CONTINUE_TIMESTEPS = int(os.environ.get("CONTINUE_TIMESTEPS", "1000000"))
    SHAPING_COEF = float(os.environ.get("SHAPING_COEF", "1.0"))

    # Windows: ensure 'spawn' start method
    if platform.system() == "Windows":
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    print(f"Creating {N_PARALLEL} env instances (shaping_coef={SHAPING_COEF})...")
    env_fns = [make_env(i, n_agents=N_AGENTS, seed=SEED, forecast_horizon=forecast_horizon,
                        shaping_coef=SHAPING_COEF,
                        battery_capacity_kwh=float(os.environ.get("BATTERY_CAPACITY_KWH", "150.0")),
                        battery_max_charge_kw=float(os.environ.get("BATTERY_MAX_CHARGE_KW", "80.0")))
               for i in range(N_PARALLEL)]

    try:
        vec_env = SubprocVecEnv(env_fns)
    except Exception as e:
        print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv (single-process).")
        vec_env = DummyVecEnv(env_fns)

    if os.path.exists(VECSTAT):
        print("Attempting to load VecNormalize stats...")
        try:
            vec_env = VecNormalize.load(VECSTAT, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print("Successfully loaded VecNormalize stats.")
        except (AssertionError, ValueError, EOFError) as e:
            print(f"Warning: Could not load VecNormalize stats ({e}), creating new VecNormalize wrapper.")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        print("No VecNormalize stats found. Creating new VecNormalize wrapper...")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Loading PPO model with matching environment...")
    model = PPO.load(checkpoint, env=vec_env)
    model.set_env(vec_env)
    print(f"Using n_steps={model.n_steps} from saved model")

    print("Continuing training for", CONTINUE_TIMESTEPS, "timesteps...")
    model.learn(total_timesteps=CONTINUE_TIMESTEPS, reset_num_timesteps=True)

    final_path = os.path.join(MODEL_DIR, "ppo_energy_continued_parallel.zip")
    print("Saving model to", final_path)
    model.save(final_path)
    try:
        vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    except Exception as e:
        print("Warning: VecNormalize save failed:", e)
    print("Done.")
