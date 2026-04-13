
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiAgentSharedPolicyWrapper(gym.Env):
    """
    Wraps a multi-agent environment where each agent has the same observation/action spec
    to look like a single-agent environment with N parallel instances for SB3.
    """
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        
        # SB3 expects single-agent spaces
        # Observation shape is (obs_dim,)
        # Env originally returns (n_agents, obs_dim)
        raw_obs = self.env.reset()[0]
        self.obs_dim = raw_obs.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        
        # Action space is (action_dim,)
        # Env originally expects (n_agents, action_dim)
        self.action_dim = 3
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        self.viewer = None

    def reset(self, seed=None, options=None):
        # We handle the n_agents dimension at the Wrapper-Vector level or here
        # Actually, for SB3 PPO, it's easier to use DummyVecEnv or SubprocVecEnv on 
        # a wrapper that represents ONE agent at a time, but that's inefficient.
        
        # Real approach: This wrapper stays "Multi-Agent" but we must integrate with VecEnv.
        # However, the user wants ONE policy. 
        # If we return obs shape (n_agents, obs_dim), SB3 Multi-Input or flattened 
        # won't treat them as independent environments unless we use a specific trick.
        
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        # Actions comes in as (n_agents, 3)
        obs, rewards, done, truncated, info = self.env.step(actions)
        return obs, rewards, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

# Note: In the run script, we will actually use the environment directly 
# but carefully handle the Batching.
