# resume_parallel.py
import os
import glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from energy_env_improved import EnergyMarketEnv

def make_env(seed_offset=0, n_agents=4, seed=123, forecast_horizon=0, shaping_coef=1.0,
             battery_capacity_kwh=50.0, battery_max_charge_kw=25.0, per_agent_max_kw=120.0,
             fairness_coeff=0.5):
    def _init():
        env = EnergyMarketEnv(n_agents=n_agents,
                              max_line_capacity_kw=200.0,
                              per_agent_max_kw=per_agent_max_kw,
                              base_price=0.12,
                              price_slope=0.002,
                              overload_multiplier=25.0,
                              forecast_horizon=forecast_horizon,
                              shaping_coef=shaping_coef,
                              battery_capacity_kwh=battery_capacity_kwh,
                              battery_max_charge_kw=battery_max_charge_kw,
                              fairness_coeff=fairness_coeff,
                              seed=seed + seed_offset)
        return Monitor(env)
    return _init

if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Search checkpoint
    candidates = glob.glob(os.path.join(MODEL_DIR, "ppo_energy_checkpoint_*.zip")) + \
                 glob.glob(os.path.join(MODEL_DIR, "ppo_energy_continued*.zip")) + \
                 glob.glob(os.path.join(MODEL_DIR, "ppo_energy_*.zip"))
    checkpoint = max(candidates, key=os.path.getmtime) if candidates else None
    if checkpoint:
        print("Resuming from checkpoint:", checkpoint)
    else:
        print("No checkpoint found in models/. Starting new training run.")

    # env vars / config
    CONTINUE_TIMESTEPS = int(os.environ.get("CONTINUE_TIMESTEPS", "1000000"))
    N_PARALLEL = int(os.environ.get("N_PARALLEL", "4"))
    SEED = int(os.environ.get("SEED", "123"))
    N_AGENTS = int(os.environ.get("N_AGENTS", "4"))
    FORECAST_HORIZON = int(os.environ.get("FORECAST_HORIZON", "0"))
    SHAPING_COEF = float(os.environ.get("SHAPING_COEF", "1.0"))
    BATTERY_CAPACITY_KWH = float(os.environ.get("BATTERY_CAPACITY_KWH", "50.0"))
    BATTERY_MAX_CHARGE_KW = float(os.environ.get("BATTERY_MAX_CHARGE_KW", "25.0"))
    PER_AGENT_MAX_KW = float(os.environ.get("PER_AGENT_MAX_KW", "120.0"))
    FAIRNESS_COEFF = float(os.environ.get("FAIRNESS_COEFF", "0.5"))
    CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", "100000"))
    CHECKPOINT_PREFIX = os.environ.get("CHECKPOINT_PREFIX", "ppo_energy_checkpoint")

    print(f"Training config: timesteps={CONTINUE_TIMESTEPS} n_parallel={N_PARALLEL} shaping={SHAPING_COEF} fairness={FAIRNESS_COEFF}")

    print(f"Creating SubprocVecEnv with {N_PARALLEL} processes...")
    env_fns = [make_env(i, n_agents=N_AGENTS, seed=SEED, forecast_horizon=FORECAST_HORIZON,
                        shaping_coef=SHAPING_COEF, battery_capacity_kwh=BATTERY_CAPACITY_KWH,
                        battery_max_charge_kw=BATTERY_MAX_CHARGE_KW, per_agent_max_kw=PER_AGENT_MAX_KW,
                        fairness_coeff=FAIRNESS_COEFF)
               for i in range(N_PARALLEL)]
    vec_env = SubprocVecEnv(env_fns)

    VECSTAT = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(VECSTAT):
        print("Attempting to load VecNormalize stats...")
        try:
            vec_env = VecNormalize.load(VECSTAT, vec_env)
            print("Successfully loaded VecNormalize stats.")
        except (AssertionError, ValueError) as e:
            print(f"Warning: Could not load VecNormalize stats ({e}). Creating new VecNormalize wrapper...")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        print("No VecNormalize stats found. Creating new VecNormalize wrapper...")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Load or create model. Use improved default hyperparams but preserve loading if checkpoint present.
    if checkpoint:
        print("Loading PPO model from checkpoint with matching env...")
        model = PPO.load(checkpoint, env=vec_env)
        model.set_env(vec_env)
        print(f"Loaded model n_steps={model.n_steps}, lr={model.learning_rate}")
    else:
        print("Creating new PPO model with tuned hyperparameters.")
        # tuned defaults (you can change via code)
        policy_kwargs = dict()  # leave empty or add net_arch if desired
        model = PPO("MlpPolicy", vec_env,
                    verbose=1,
                    learning_rate=float(os.environ.get("PPO_LR", "1e-4")),
                    n_steps=int(os.environ.get("PPO_N_STEPS", "4096")),
                    batch_size=int(os.environ.get("PPO_BATCH_SIZE", "1024")),
                    n_epochs=int(os.environ.get("PPO_N_EPOCHS", "10")),
                    gamma=float(os.environ.get("PPO_GAMMA", "0.99")),
                    ent_coef=float(os.environ.get("PPO_ENT_COEF", "0.0")),
                    tensorboard_log=os.path.join(MODEL_DIR, "tb"),
                    policy_kwargs=policy_kwargs)
        init_path = os.path.join(MODEL_DIR, "ppo_energy_initial.zip")
        model.save(init_path)
        print("Saved initial model at", init_path)

    # checkpoint callback
    checkpoint_cb = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=MODEL_DIR, name_prefix=CHECKPOINT_PREFIX)

    print("Continuing training for", CONTINUE_TIMESTEPS, "timesteps...")
    model.learn(total_timesteps=CONTINUE_TIMESTEPS, reset_num_timesteps=True, callback=checkpoint_cb)

    final_path = os.path.join(MODEL_DIR, "ppo_energy_continued_parallel.zip")
    print("Saving final model to", final_path)
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print("Done.")
