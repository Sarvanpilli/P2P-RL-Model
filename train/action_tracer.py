
import time
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

class ActionTracer:
    """
    Records the decision lifecycle of the Autonomous System.
    
    Trace Path:
    Observation -> RL Intent -> Optimized Plan -> Safety Check -> Final Execution
    
    This ensures that for every action taken, we know:
    1. What the RL wanted to do (Strategic Intent).
    2. How the Optimizer adjusted it (Feasibility).
    3. If the Safety Supervisor intervened (Veto/Fallback).
    """
    
    def __init__(self, history_len: int = 1024):
        self.history_len = history_len
        self.trace_buffer = deque(maxlen=history_len)
        self.incident_log = []

    def log_decision(self, 
                     step: int,
                     agent_id: int,
                     obs: np.ndarray,
                     rl_intent: np.ndarray,
                     optimized_action: np.ndarray,
                     final_action: np.ndarray,
                     safety_status: str,
                     metrics: Dict[str, Any]):
        """
        Logs a single decision event.
        
        Args:
            step: Current timestep.
            agent_id: ID of the agent.
            obs: Raw observation vector.
            rl_intent: Raw action output by RL policy.
            optimized_action: Action output by Layer 2 (Optimizer).
            final_action: Action executed after Layer 3 (Safety).
            safety_status: 'OK', 'OPTIMIZED', 'VETOED', 'FALLBACK'.
            metrics: Dict of health metrics (e.g., OOD score).
        """
        record = {
            "timestamp": time.time(),
            "step": step,
            "agent": agent_id,
            "obs_summary": {
                "soc": float(obs[1]), # Assuming idx 1 is SoC
                "demand": float(obs[0]),
                "pw": float(obs[2])
            },
            "intent": rl_intent.tolist(),
            "optimized": optimized_action.tolist(),
            "final": final_action.tolist(),
            "status": safety_status,
            "metrics": metrics
        }
        
        self.trace_buffer.append(record)
        
        if safety_status in ['VETOED', 'FALLBACK']:
            self.incident_log.append(record)
            
    def get_recent_traces(self) -> List[Dict]:
        return list(self.trace_buffer)

    def get_incidents(self) -> List[Dict]:
        return self.incident_log
