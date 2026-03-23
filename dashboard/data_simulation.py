# dashboard/data_simulation.py
import math
import random

def generate_step(step):
    """Physics-based simulation of peak solar/wind and demand cycles."""
    hour = step % 24
    
    # Cycles
    solar = Math_sin_capped(hour, 6, 18, 12) * 0.8
    wind = 0.3 + 0.2 * math.sin(hour * math.PI / 12 + 1)
    
    # Generation (Solar, Wind, Hybrid, Std)
    gen = [
        solar,
        wind * 0.5,
        solar * 0.3,
        solar * 0.4
    ]
    
    # Demand (Peak hours 17-21)
    demand = [
        0.4 + 0.3 * math.sin((hour - 8) * math.PI / 12 + 0.5),
        0.3 + 0.2 * math.cos(hour * 0.4),
        0.2 + (0.6 if 17 <= hour <= 21 else 0.1),
        0.5 + 0.2 * math.sin(hour * 0.3)
    ]
    
    surplus = [g - d for g, d in zip(gen, demand)]
    
    sellers_count = sum(1 for s in surplus if s > 0.05)
    buyers_count  = sum(1 for s in surplus if s < -0.05)
    
    # P2P Matching
    if sellers_count > 0 and buyers_count > 0:
        p2p = min(sellers_count, buyers_count) * (0.15 + random.random() * 0.10)
        price = 0.20 + random.random() * 0.15
    else:
        p2p = 0.0
        price = 0.10
        
    # State of Charge (SoC) estimation
    soc = [
        max(0.2, min(0.8, 0.4 + 0.4 * math.sin((hour - 8) * math.PI / 12))),
        max(0.3, min(0.9, 0.5 + 0.3 * math.cos(hour * 0.2))),
        0.3 + (hour - 17) * 0.08 if hour >= 17 else (0.7 if hour < 8 else 0.3),
        0.45 + 0.15 * math.sin(hour * math.PI / 16)
    ]
    
    return {
        "step": step,
        "hour": hour,
        "gen": gen,
        "demand": demand,
        "surplus": surplus,
        "sellers": sellers_count,
        "buyers": buyers_count,
        "p2p": p2p,
        "price": price,
        "soc": soc
    }

def Math_sin_capped(val, start, end, period):
    if val < start or val > end:
        return 0
    return max(0, math.sin((val - start) * math.PI / (end - start)))

def get_agent_role(surplus):
    if surplus > 0.05:  return {"label": 'Selling', "color": 'green'}
    if surplus < -0.05: return {"label": 'Buying',  "color": 'blue'}
    return {"label": 'Idle', "color": 'gray'}
