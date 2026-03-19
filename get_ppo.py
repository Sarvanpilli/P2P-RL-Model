import inspect
from stable_baselines3 import PPO
import os

source = inspect.getsource(PPO.train)
with open("ppo_train_source.py", "w", encoding="utf-8") as f:
    f.write(source)
