import inspect
import os
import sys

# Ensure current directory is in path (it usually is)
sys.path.insert(0, os.getcwd())

try:
    from train.reward_tracker import RewardTracker
    print(f"Loaded from: {inspect.getfile(RewardTracker)}")
    sig = inspect.signature(RewardTracker.calculate_total_reward)
    print(f"Signature: {sig}")
    
    # Check for v2g_bonus explicitly
    if 'v2g_bonus' in sig.parameters:
        print("v2g_bonus PRESENT")
    else:
        print("v2g_bonus MISSING")
        
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
