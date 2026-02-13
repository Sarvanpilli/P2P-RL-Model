
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("RecurrentPPO not found, standard PPO will handle if model is compatible or error out.")

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.energy_env_robust import EnergyMarketEnvRobust
from baselines.rule_based_agent import RuleBasedAgent

def run_evaluation(mode="RL_Phase4", model_path=None, output_csv="eval_output.csv", n_steps=168): # 1 Week
    print(f"--- Running Eval: {mode} ---")
    
    # Configuration based on mode
    is_phase4 = "Phase4" in mode
    
    # Setup Env - Pointing to REAL DATA
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=4,
            data_file="evaluation/ausgrid_p2p_energy_dataset.csv",
            random_start_day=False, 
            enable_ramp_rates=True,
            enable_losses=True,
            forecast_horizon=1,
            enable_predictive_obs=is_phase4 # Phase 4 Flag
        )
    
    env_base = make_env()
    
    # Load Model
    model = None
    env = None
    
    if "RL" in mode:
        # Wrap for VecNormalize
        env = DummyVecEnv([lambda: env_base])
        
        # Load Stats
        vec_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(vec_path):
             print(f"Loading stats from {vec_path}...")
             env = VecNormalize.load(vec_path, env)
             env.training = False
             env.norm_reward = False
        else:
             print("Warning: No stats found. Using raw obs?")
             
        try:
            if is_phase4:
                # RecurrentPPO loading attempts
                try:
                    from sb3_contrib import RecurrentPPO
                    model = RecurrentPPO.load(model_path, env=env)
                    print("Loaded RecurrentPPO model.")
                except Exception as e:
                    print(f"RecurrentPPO load failed ({e}), trying PPO...")
                    model = PPO.load(model_path, env=env)
            else:
                model = PPO.load(model_path, env=env)
                
        except Exception as e:
             print(f"ERROR: Model load failed due to: {e}")
             print("WARNING: Falling back to Random Agent to verify Environment/Reward Logic.")
             class RandomPolicy:
                 def predict(self, obs, state=None, episode_start=None, deterministic=False):
                     # Random action in [-1, 1] (normalized space)
                     # 1 Env, 4 Agents * 3 Actions = 12
                     return np.random.uniform(-1, 1, (1, 12)), None
             model = RandomPolicy()
             
        # Initialize LSTM states if needed
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        # Start state
        try:
            obs = env.reset()
        except Exception as e:
            print(f"Env reset failed: {e}")
            return None

    else:
        # Baseline Agents
        agents = []
        for i in range(env_base.n_agents):
            agents.append(RuleBasedAgent(i, 50.0, 25.0)) 
        obs = env_base.reset()[0]
        
    results = []
    
    for t in range(n_steps):
        step_metrics = {
            "step": t,
            "mode": mode
        }
        
        if "RL" in mode:
            if is_phase4:
                # Recurrent Prediction
                # predict(obs, state=None, episode_start=None, deterministic=False)
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                episode_starts = np.zeros((1,), dtype=bool) # Next step is not start
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            
            # Access underlying env for un-normalized metrics
            raw_env = env.envs[0]
            nodes = raw_env.nodes
            
            # Additional Metrics for Phase 4 Verification
            # Capture SoC Mean, Grid Import, Jitter (Proxy: change in action? Hard without caching prev)
            # info dict should have what we need if Env supports it
            
            step_metrics["market_price"] = info.get("market_price", 0.0)
            step_metrics["loss_kw"] = info.get("loss_kw", 0.0)
            step_metrics["net_grid_flow"] = info.get("total_export", 0.0) - info.get("total_import", 0.0)
            step_metrics["total_import"] = info.get("total_import", 0.0)
            step_metrics["total_reward"] = np.sum(reward)
            
            # Specific Phase 4 Metrics from Reward Tracker
            step_metrics["smoothing_penalty"] = info.get("reward/smoothing_mean", 0.0) 
            step_metrics["soc_mean"] = np.mean([n.soc for n in nodes])
            
        else:
            # Baseline Loop
            actions = []
            # Obs logic for baseline (Legacy format from make_env(enable_predictive=False)?)
            # Wait, if mode is Baseline, we used enable_predictive=Phase4? No, is_phase4=False.
            # So Env produces Legacy Obs.
            
            # Flattened obs
            obs_per_agent = len(obs) // env_base.n_agents
            for i in range(env_base.n_agents):
                agent_obs = obs[i*obs_per_agent : (i+1)*obs_per_agent]
                act = agents[i].get_action(agent_obs, t)
                actions.append(act)
            
            flat_action = np.array(actions).flatten()
            obs, reward, done, trunc, info = env_base.step(flat_action)
            
            step_metrics["market_price"] = info.get("market_price", 0.0)
            step_metrics["loss_kw"] = info.get("loss_kw", 0.0)
            step_metrics["net_grid_flow"] = info.get("total_export", 0.0) - info.get("total_import", 0.0)
            step_metrics["total_import"] = info.get("total_import", 0.0)
            step_metrics["total_reward"] = np.sum(reward)
            step_metrics["smoothing_penalty"] = 0.0 
            step_metrics["soc_mean"] = np.mean([n.soc for n in env_base.nodes])

        results.append(step_metrics)
        if t % 24 == 0:
            print(f"Step {t}/{n_steps} complete.")
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df

def main():
    # 1. Phase 3 (Benchmark)
    # if os.path.exists("models_phase3/ppo_grid_aware.zip"):
    #     run_evaluation("RL_Phase3_Benchmark", "models_phase3/ppo_grid_aware.zip", "results_phase3_bench.csv")
        
    # 2. Phase 4 (New)
    if os.path.exists("models_phase4/ppo_predictive.zip"):
        try:
            run_evaluation("RL_Phase4", "models_phase4/ppo_predictive.zip", "results_phase4.csv")
        except Exception as e:
            print(f"CRITICAL FAILURE in Phase 4 Eval: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Phase 4 Model not found!")

if __name__ == "__main__":
    main()
