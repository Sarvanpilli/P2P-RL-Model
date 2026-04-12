
import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO
from typing import List

# Custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from train.energy_env_robust import EnergyMarketEnvRobust

app = FastAPI(title="SLIM Real-Time P2P Inference API")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class StateRequest(BaseModel):
    observation: List[float]

class ActionResponse(BaseModel):
    action: List[float]

# Global variables for model and env
model = None
env = None
vec_norm = None
model_loaded = False

def lazy_load_model():
    global model, env, vec_norm, model_loaded
    if model_loaded: return
    
    print("Initializing Real-Time Inference Model (Lazy)...")
    # Path to the BEST model found during staged optimization
    model_path = "models_staged/best_model.zip"
    vec_path = "models_staged/best_vec_normalize.pkl"
    
    try:
        # 1. Base Environment
        env_raw = EnergyMarketEnvRobust(n_agents=4, data_file="processed_hybrid_data.csv")
        env = DummyVecEnv([lambda: env_raw])
        
        # 2. Model & Normalization
        if os.path.exists(model_path) and os.path.exists(vec_path):
            vec_norm = VecNormalize.load(vec_path, env)
            vec_norm.training = False # Ensure no further training stats are collected
            vec_norm.norm_reward = False
            
            model = PPO.load(model_path)
            model_loaded = True
            print("Optimized Model and Normalization loaded successfully.")
        else:
            print(f"WARNING: Model ({model_path}) or VecPath ({vec_path}) not found.")
    except Exception as e:
        print(f"ERROR loading model: {e}")

@app.get("/")
def home():
    return {"status": "online", "system": "SLIM v3", "model_updated": "Phase 4 Optimized", "model_loaded": model_loaded}

@app.post("/predict", response_model=ActionResponse)
def predict(request: StateRequest):
    lazy_load_model()
    
    if model_loaded:
        # Observations MUST be normalized before prediction
        obs = np.array(request.observation).astype(np.float32)
        if obs.shape[0] != 104:
            raise HTTPException(status_code=400, detail=f"Observation mismatch. Expected 104, got {obs.shape[0]}")
            
        # Apply normalization from the training phase
        obs_norm = vec_norm.normalize_obs(obs)
        
        action, _ = model.predict(obs_norm, deterministic=True)
        return {"action": action.tolist()}
    else:
        # Fallback to random if model not loaded
        raw_env = EnergyMarketEnvRobust(n_agents=4, data_file="processed_hybrid_data.csv")
        random_action = raw_env.action_space.sample()
        return {"action": random_action.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
