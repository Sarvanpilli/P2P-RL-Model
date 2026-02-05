
import sys
import os
import numpy as np

print(f"NumPy Version: {np.__version__}")

# Apply the same patch as dashboard.py to test it
try:
    import numpy.core
    if 'numpy._core' not in sys.modules:
        print("Patching numpy._core...")
        sys.modules['numpy._core'] = numpy.core
    if 'numpy._core.numeric' not in sys.modules:
        from numpy.core import numeric
        sys.modules['numpy._core.numeric'] = numeric
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except ImportError as e:
    print(f"Patch failed: {e}")

try:
    from train.energy_env_robust import EnergyMarketEnvRobust
except ImportError:
    # Fix path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train.energy_env_robust import EnergyMarketEnvRobust

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

MODEL_PATH = "models/ppo_energy_final.zip"

def test_load():
    print("--- 1. Testing Env Creation ---")
    try:
        env = EnergyMarketEnvRobust(n_agents=4, data_file="test_day_profile.csv")
        print("Env created successfully.")
    except Exception as e:
        print(f"FAILED to create env: {e}")
        return

    print("\n--- 2. Testing VecNormalize Load ---")
    vecnorm_path = "models/vec_normalize.pkl"
    if os.path.exists(vecnorm_path):
        try:
            # We need to wrap env in DummyVecEnv for VecNormalize? No, VecNormalize.load expects venv.
            from stable_baselines3.common.vec_env import DummyVecEnv
            venv = DummyVecEnv([lambda: env])
            venv = VecNormalize.load(vecnorm_path, venv)
            print("VecNormalize loaded successfully.")
        except Exception as e:
            print(f"FAILED to load VecNormalize: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("VecNormalize file not found.")

    print("\n--- 3. Testing PPO Model Load ---")
    if os.path.exists(MODEL_PATH):
        try:
            model = PPO.load(MODEL_PATH)
            print("PPO Model loaded successfully.")
        except Exception as e:
            print(f"FAILED to load PPO Model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model file {MODEL_PATH} not found.")

if __name__ == "__main__":
    test_load()
