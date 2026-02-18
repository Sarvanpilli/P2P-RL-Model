
import os
import subprocess
import time

def run_eval(n_agents, model_path, no_safety=False, no_p2p=False):
    print(f"\n>>> Running Evaluation: N={n_agents}, Safety={not no_safety}, P2P={not no_p2p}")
    
    cmd = [
        "python", "research_q1/novelty/evaluate_slim_scale.py",
        "--n_agents", str(n_agents),
        "--model", model_path,
        "--episodes", "10"
    ]
    
    if no_safety: cmd.append("--no_safety")
    if no_p2p: cmd.append("--no_p2p")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model_path}: {e}")

if __name__ == "__main__":
    print("=== Starting Batch Evaluation ===")
    
    # 1. Scalability N=10
    # Expected Path: research_q1/models/slim_ppo_N10/slim_ppo_final.zip
    run_eval(10, "research_q1/models/slim_ppo_N10/slim_ppo_final.zip")
    
    # 2. Ablation: No Safety
    # Expected Path: research_q1/models/slim_ablation_N4_NoSafety/final_model.zip
    run_eval(4, "research_q1/models/slim_ablation_N4_NoSafety/final_model.zip", no_safety=True)
    
    # 3. Ablation: No P2P
    # Expected Path: research_q1/models/slim_ablation_N4_NoP2P/final_model.zip
    run_eval(4, "research_q1/models/slim_ablation_N4_NoP2P/final_model.zip", no_p2p=True)
    
    print("\n=== Batch Evaluation Complete ===")
