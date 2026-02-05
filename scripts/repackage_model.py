
import sys
import types
import numpy as np

# Mock numpy._core to point to numpy.core
# This effectively redirects the pickle loader to the old location
try:
    import numpy.core
    # Create the _core module
    core_mock = types.ModuleType("numpy._core")
    # Map attributes
    core_mock.numeric = numpy.core.numeric
    core_mock.multiarray = numpy.core.multiarray
    core_mock.umath = numpy.core.umath
    
    # Inject into sys.modules
    sys.modules["numpy._core"] = core_mock
    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.umath"] = numpy.core.umath
    
    print("[Patch] Successfully injected numpy._core mock for compatibility.")
except Exception as e:
    print(f"[Patch] Failed to inject mock: {e}")

# Now import the rest
from stable_baselines3 import PPO
import os

model_path = "models/ppo_energy_final.zip"
target_path = "models/ppo_energy_final_compatible.zip"

print(f"Attempting to load {model_path}...")
try:
    model = PPO.load(model_path, custom_objects={})
    print("Success! Model loaded.")
    
    print(f"Saving compatible version to {target_path}...")
    model.save(target_path)
    print("Done.")
    
except Exception as e:
    print(f"Failed: {e}")
    # Print detailed traceback
    import traceback
    traceback.print_exc()
