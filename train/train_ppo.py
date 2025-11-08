# train/train_ppo.py
"""
Updated training script with Monitor, TensorBoard logging, seed control, and optional baseline training mode.
Run as module: python -m train.train_ppo --help
"""
import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.multi_p2p_env import MultiP2PEnergyEnv

def make_env_fn(n_agents, episode_len, seed, config=None):
    def _init():
        # MultiP2PEnergyEnv signature may accept config/kwargs — pass as appropriate
        env = MultiP2PEnergyEnv(n_agents=n_agents, episode_len=episode_len, seed=seed, config=config)
        env = Monitor(env)
        return env
    return _init

def run_training(args):
    os.makedirs(args.save_dir, exist_ok=True)

    env_fns = [make_env_fn(args.n_agents, args.episode_len, seed=i + args.seed, config=None)
               for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    model = PPO('MlpPolicy', vec_env,
                verbose=1,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                gamma=args.gamma,
                seed=args.seed,
                tensorboard_log=(args.tensorboard_log if args.tensorboard_log else None))

    print(f"Training PPO for {args.total_timesteps} timesteps on {args.n_agents} agents (centralized policy)")
    model.learn(total_timesteps=args.total_timesteps)
    model_path = os.path.join(args.save_dir, 'ppo_p2p.zip')
    model.save(model_path)
    print(f"Saved model to {model_path}")

    return model_path

def quick_eval(model_path, n_eval_episodes, n_agents, episode_len, deterministic=True):
    # Quick evaluation printing per-step summary (useful for spot-checks)
    model = PPO.load(model_path)
    env = MultiP2PEnergyEnv(n_agents=n_agents, episode_len=episode_len, seed=0)
    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            # info keys depend on your env implementation. Adjust prints as needed.
            print(f"t={getattr(env,'t',None)}, reward={float(reward):.4f}, clearing_price={info.get('clearing_price')}, gini={info.get('gini')}")
        print("episode finished\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=6)
    parser.add_argument('--episode_len', type=int, default=24)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--total_timesteps', type=int, default=200_000)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--tensorboard_log', type=str, default='tb_logs')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_episodes', type=int, default=1)
    args = parser.parse_args()

    if args.eval_only:
        model_path = os.path.join(args.save_dir, 'ppo_p2p.zip')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        quick_eval(model_path, args.eval_episodes, args.n_agents, args.episode_len)
    else:
        run_training(args)
