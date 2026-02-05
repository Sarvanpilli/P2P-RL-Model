from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import pickle
import os

def check_shapes():
    model_path = "models/ppo_energy_final.zip"
    vecnorm_path = "models/vec_normalize.pkl"
    
    if os.path.exists(model_path):
        try:
            # We can load model without env to check its internal shapes
            model = PPO.load(model_path)
            print(f"Model Observation Space: {model.observation_space}")
            print(f"Model Action Space: {model.action_space}")
        except Exception as e:
            print(f"Error loading model: {e}")

    if os.path.exists(vecnorm_path):
        try:
            with open(vecnorm_path, "rb") as f:
                data = pickle.load(f)
            # vec_normalize stores 'obs_rms' which has 'mean' of shape (dim,)
            obs_dim = data.obs_rms.mean.shape[0]
            print(f"VecNormalize Expected Obs Dim: {obs_dim}")
        except Exception as e:
            print(f"Error loading vecnorm: {e}")

if __name__ == "__main__":
    check_shapes()
