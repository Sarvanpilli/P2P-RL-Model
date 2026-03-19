import sys
import os

sys.path.insert(0, os.getcwd())

from research_q1.env.reward_tracker import RewardTracker
import numpy as np

rt = RewardTracker(n_agents=4)
reward = rt.calculate_total_reward(
    profits=np.zeros(4),
    grid_import_penalties=np.zeros(4),
    soc_penalties=np.zeros(4),
    grid_overload_costs=np.zeros(4),
    battery_costs=np.zeros(4),
    export_penalties=np.zeros(4),
    p2p_volume_kwh=10.0
)
print(f"Density reward with p2p=10 kWh -> total reward = {reward:.4f}")
assert abs(reward - 0.5) < 0.01, f"Expected ~0.5, got {reward}"
info = rt.get_info()
print(f"density_reward key: {info['density_reward']:.4f}")
print("Phase A RewardTracker: PASS")
