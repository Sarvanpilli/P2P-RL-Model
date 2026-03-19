import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_q1.env.energy_env_robust import EnergyMarketEnvRobust
from research_q1.novelty.lagrangian_manager import LagrangianManager
from research_q1.novelty.lagrangian_ppo import LagrangianPPO
from research_q1.novelty.gnn_policy import CTDEGNNPolicy

class LagrangianUpdateCallback(BaseCallback):
    """
    Hooks into the PPO training loop to extract constraint violations from
    the environment and update the Dual Multipliers (lambdas).
    """
    def __init__(self, lagrangian_manager, verbose=0):
        super().__init__(verbose)
        self.lag_manager = lagrangian_manager
        
    def _on_step(self) -> bool:
        # We process violations at the end of the rollout, so we just collect stats if needed,
        # but stable_baselines3 environments already return "info" dicts on step.
        return True

    def _on_rollout_end(self) -> None:
        """
        Executed after gathering `n_steps` of interactions.
        Pulls the violation metrics from the environment's `info` buffer to update lambdas.
        """
        # In VecEnvs, we must inspect the environment's recent infos.
        # However, for simplicity and mathematical consistency, we can extract
        # the rolling average violations directly if the env stores them, or 
        # from the rollout buffer if we injected them.
        
        # We access the most recent info dicts from the VecEnv
        infos = self.locals.get("infos", [])
        if not infos:
            return
            
        # Average violations over the batch of recent infos
        batch_soc_v = []
        batch_line_v = []
        batch_volt_v = []
        
        for info in infos:
            if "violation_dict" in info:
                v = info["violation_dict"]
                batch_soc_v.append(v.get("soc", 0.0))
                batch_line_v.append(v.get("line", 0.0))
                batch_volt_v.append(v.get("voltage", 0.0))
                
        if batch_soc_v:
            mean_soc = np.mean(batch_soc_v)
            mean_line = np.mean(batch_line_v)
            mean_volt = np.mean(batch_volt_v)
            
            # 1. Update Dual Variables
            self.lag_manager.update(mean_soc, mean_line, mean_volt)
            
            # 2. Extract new lambdas
            lambdas = self.lag_manager.get_lambdas()
            
            # 3. Log to TensorBoard
            self.logger.record("lagrangian/lambda_soc", lambdas["lambda_soc"])
            self.logger.record("lagrangian/lambda_line", max(0.0, mean_line)) # Just logging line violation
            
            # 4. Push new State-Path Lambdas down into the Environment
            # For a DummyVecEnv, env_method calls the function on all sub-environments
            try:
                self.training_env.env_method(
                    "set_lambdas", 
                    lambda_soc=lambdas["lambda_soc"],
                    lambda_line=lambdas["lambda_line"],  # Handled by LagrangianPPO mostly
                    lambda_voltage=lambdas["lambda_voltage"]
                )
            except Exception as e:
                # Fallback if standard env wrapping blocks it
                if hasattr(self.training_env.envs[0], 'set_lambdas'):
                    for sub_env in self.training_env.envs:
                        sub_env.set_lambdas(
                            lambdas["lambda_soc"], 
                            lambdas["lambda_line"], 
                            lambdas["lambda_voltage"]
                        )


def main():
    parser = argparse.ArgumentParser(description="Master Training Script for P2P-RL Dual-Path GATv2")
    parser.add_argument("--ablation", type=str, choices=["baseline", "gnn_hard", "gnn_lagrangian"], 
                        default="gnn_lagrangian", help="Select the model architecture to run.")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps.")
    parser.add_argument("--n_agents", type=int, default=4, help="Number of nodes in the Microgrid.")
    args = parser.parse_args()

    print(f"\n==========================================")
    print(f" INITIALIZING TRAINING RUN: {args.ablation.upper()}")
    print(f"==========================================\n")

    # 1. Initialize the Environment
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=args.n_agents,
            data_file="processed_hybrid_data.csv",
            random_start_day=True,
            enable_ramp_rates=True,
            enable_losses=True
        )

    vec_env = DummyVecEnv([make_env])
    # Normalize rewards for PPO stability, but keep observations raw if GNN needs specific physics parsing
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=100.0)

    # 2. Setup Run Specifics
    tensorboard_log = f"./research_q1/results/tb_logs/{args.ablation}"
    os.makedirs(tensorboard_log, exist_ok=True)
    
    callbacks = []

    if args.ablation == "baseline":
        # Run 1: Original MLP + Hard Guard
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device="auto"
        )
        
    elif args.ablation == "gnn_hard":
        # Run 2: GNN + Hard Guard
        model = PPO(
            CTDEGNNPolicy,
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device="auto",
            policy_kwargs=dict(n_agents=args.n_agents)
        )
        
    elif args.ablation == "gnn_lagrangian":
        # Run 3: GNN + Dual-Path Lagrangian Safety
        lag_manager = LagrangianManager(
            lr=1e-3, 
            damping=1e-4, 
            max_lambda=10.0
        )
        
        # Use our custom LagrangianPPO to inject Differentiable Path Action penalties
        model = LagrangianPPO(
            CTDEGNNPolicy,
            vec_env,
            lagrangian_manager=lag_manager,
            action_limit=1.0, # Box space is [-1, 1], so limit is 1.0 boundary
            verbose=1,
            tensorboard_log=tensorboard_log,
            device="auto",
            policy_kwargs=dict(n_agents=args.n_agents)
        )
        
        lag_callback = LagrangianUpdateCallback(lagrangian_manager=lag_manager)
        callbacks.append(lag_callback)

    # 3. Execute Training
    print(f"Commencing Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="run")
    
    # 4. Save Artifacts
    model_path = f"research_q1/results/models/{args.ablation}_{args.n_agents}node"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
