import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research_q1.novelty.slim_env import EnergyMarketEnvSLIM

class MarketLoggerCallback(BaseCallback):
    """Callback to extract explicitly requested metrics from the environment's info dict to Tensorboard."""
    def __init__(self, verbose=0):
        super(MarketLoggerCallback, self).__init__(verbose)
        
    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "p2p_volume" in info:
                # Log critical market metrics to track agent strategy evolution
                self.logger.record("market/p2p_volume", info["p2p_volume"])
                self.logger.record("market/grid_import", info["grid_import"])
                self.logger.record("market/grid_export", info["grid_export"])
                self.logger.record("market/delta", info["delta"])
                self.logger.record("market/seller_price", info["seller_price"])
                self.logger.record("market/buyer_price", info["buyer_price"])
                self.logger.record("reward/profit", info["profit"])
        return True

def train(timesteps=100_000, seed=42):
    LOG_DIR = f"research_q1/models/rl_market_seed_{seed}"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    def make_env():
        return EnergyMarketEnvSLIM(
            n_agents=4, 
            data_file="fixed_training_data.csv",
            enable_safety=True,
            enable_p2p=True,
            seed=seed
        )
        
    env = DummyVecEnv([make_env])
    # Very important to normalize the 26-dimensional obs space with the new P2P features
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.05,
        verbose=1,
        tensorboard_log="research_q1/logs/tb_rl_market",
        seed=seed
    )
    
    callback = MarketLoggerCallback()
    
    print(f"=== Starting RL Market Training (Seed {seed}) ===")
    model.learn(total_timesteps=timesteps, callback=callback, tb_log_name=f"PPO_DynamicMarket_{seed}")
    
    model.save(f"{LOG_DIR}/final_model")
    env.save(f"{LOG_DIR}/vec_normalize.pkl")
    print(f"Model saved completely to {LOG_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.timesteps, args.seed)
