"""
Phase 14: Extension to N=24 Scalability
=========================================
Loads the existing N=16 model and performs knowledge transfer to start 
training N=24 agents for 800,000 steps.
"""

import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.run_scalability_experiment_v5 import VectorizedMultiAgentEnv, CurriculumCallback, evaluate_scalability, generate_step_plots

# CONFIG
MODEL_DIR = "models_scalable_v5"
RESULTS_DIR = "research_q1/results/scalability_v5"
N_AGENTS = 24
TRAIN_STEPS = 800000

def run_n24_extension():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"\n===== PHASE 14 SCALABILITY EXTENSION N={N_AGENTS} =====")
    train_env = VectorizedMultiAgentEnv(n_agents=N_AGENTS)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Initialize PPO model for N=24
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
    
    # Load N=16 parameters for knowledge transfer
    n16_path = os.path.join(MODEL_DIR, "ppo_n16_v5.zip")
    if os.path.exists(n16_path):
        print(f"Loading knowledge from {n16_path}...")
        n16_model = PPO.load(n16_path)
        model.set_parameters(n16_model.get_parameters())
        print("Knowledge transfer complete.")
    else:
        print("Warning: N=16 model not found. Training from scratch.")
        
    callback = CurriculumCallback(n_agents=N_AGENTS, patience_steps=150000)
    model.learn(total_timesteps=TRAIN_STEPS, callback=callback)
    
    # Save the N=24 model and generate plots
    model.save(os.path.join(MODEL_DIR, f"ppo_n{N_AGENTS}_v5"))
    generate_step_plots(callback.history, N_AGENTS)
    
    # Evaluate and append to metrics
    res = evaluate_scalability(model, N_AGENTS)
    print("\nN=24 Evaluation Results:", res)
    
    # Update the scalability metrics file
    csv_path = os.path.join(RESULTS_DIR, "scalability_metrics_v5_extended.csv")
    df = pd.DataFrame([res])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    train_env.close()

if __name__ == "__main__":
    run_n24_extension()
