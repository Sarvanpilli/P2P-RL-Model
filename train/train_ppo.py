# train/train_ppo.py
import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.multi_p2p_env import MultiP2PEnergyEnv

def make_env_fn(n_agents, episode_len, seed):
    def _init():
        env = MultiP2PEnergyEnv(n_agents=n_agents, episode_len=episode_len, seed=seed)
        return env
    return _init

def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    env_fns = [make_env_fn(args.n_agents, args.episode_len, seed=i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    model = PPO('MlpPolicy', vec_env,
                verbose=1,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                gamma=args.gamma,
                seed=args.seed)

    total_timesteps = args.total_timesteps
    print(f"Training PPO for {total_timesteps} timesteps on {args.n_agents} agents (centralized policy)")

    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(save_dir, 'ppo_p2p.zip')
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # quick evaluation: run one episode and print info
    obs = vec_env.reset()
    done = False
    env = vec_env.envs[0]
    obs = obs[0]  # DummyVecEnv returns (n_envs, obs_dim)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # reward is now scalar (sum of all agents), per-agent rewards are in info
        reward_per_agent = info.get('rewards_per_agent', np.array([reward]))
        print(f"t={env.t}, reward={reward:.4f}, reward_per_agent_sum={reward_per_agent.sum():.4f}, clearing_price={info['clearing_price']:.3f}, gini={info['gini']:.3f}")
        if done:
            break
        # for DummyVecEnv, obs is returned as array shape (n_envs, obs_dim)
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
    print("Eval finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=5, help='Number of prosumers')
    parser.add_argument('--episode_len', type=int, default=24)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--total_timesteps', type=int, default=200_000)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    main(args)
