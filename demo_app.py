
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# P2P-RL-Model Imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train.energy_env_robust import EnergyMarketEnvRobust
from simulation.microgrid import AgentType

# --------------------------------------------------------------------------
# 1. Configuration & Setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="P2P Energy Trading Demo",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 2. Sidebar Controls
# --------------------------------------------------------------------------
st.sidebar.title("⚡ System Config")

st.sidebar.markdown("### Agents")
n_prosumers = st.sidebar.number_input("Prosumers (Solar+Batt)", value=100, disabled=True)
n_consumers = st.sidebar.number_input("Consumers (Load Only)", value=25, disabled=True)
n_agents = n_prosumers + n_consumers

st.sidebar.markdown("### Grid Constraints")
grid_capacity_target = st.sidebar.slider("Transformer Capacity (kW)", 500, 5000, 2200, step=100)

st.sidebar.markdown("### Policy Control")
policy_mode = st.sidebar.radio("Agent Brain", ["PPO (Trained)", "Random (Baseline)"])
demo_mode = st.sidebar.checkbox("Deterministic Demo Mode", value=True)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary")

# --------------------------------------------------------------------------
# 3. Main Dashboard Header
# --------------------------------------------------------------------------
st.title("Large-Scale P2P Energy Trading – Live Demo")
st.markdown(f"**{n_prosumers} Prosumers** | **{n_consumers} Consumers** | Carbon-Aware Control")

# Placeholders for Live Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    m_hour = st.empty()
with col2:
    m_price = st.empty()
with col3:
    m_grid = st.empty()
with col4:
    m_carbon = st.empty()

# Log Console
st.markdown("### 📜 Live Event Log")
log_area = st.empty()

# Plots Area
st.markdown("### 📈 Real-Time Market Analysis")
p_col1, p_col2 = st.columns(2)
with p_col1:
    plot_spot_soc = st.empty()
with p_col2:
    plot_spot_price = st.empty()

# --------------------------------------------------------------------------
# 4. Simulation Logic
# --------------------------------------------------------------------------
def run_simulation(grid_target_kw, use_ppo=True, deterministic=True):
    # Determinism
    seed = 42 if deterministic else None
    
    # 1. Environment Setup
    # Trick: Env normally scales capacity by sqrt(N). 
    # We want exact control via slider. 
    # Formula: internal_cap = input_arg * sqrt(N)
    # So: input_arg = target / sqrt(N)
    base_param = grid_target_kw / np.sqrt(n_agents)
    
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=n_agents,
            n_prosumers=n_prosumers,
            n_consumers=n_consumers,
            max_line_capacity_kw=base_param, # Will scale back to target
            timestep_hours=1.0,
            forecast_horizon=1,
            data_file="scenarios/user_scenario_data.csv",
            random_start_day=False,
            seed=seed
        )
    
    vec_env = DummyVecEnv([make_env])
    
    # 2. Check Data
    if not os.path.exists("scenarios/user_scenario_data.csv"):
        st.error("Data file missing! Run data generator first.")
        return

    # 3. Load Normalization
    stats_path = "models/vec_normalize.pkl"
    if not os.path.exists(stats_path):
        stats_path = "train/models/vec_normalize.pkl"
    
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    # 4. Load Model
    model = None
    if use_ppo:
        model_path = "models/ppo_energy_final.zip"
        if not os.path.exists(model_path):
             model_path = "train/models/ppo_energy_final.zip"
        
        if not os.path.exists(model_path):
            st.error("CRITICAL: Trained PPO model not found!")
            st.stop()
            
        model = PPO.load(model_path)
    
    # 5. Loop
    obs = vec_env.reset()
    
    logs = []
    
    history = {
        'soc': [],
        'price': [],
        'grid': [],
        'carbon': []
    }
    
    # Initial State Force (Optional for visuals)
    # Force defaults
    
    for t in range(24):
        # Action
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = [vec_env.action_space.sample()]
            
        obs, rewards, dones, info_list = vec_env.step(action)
        info = info_list[0]
        
        # Metrics
        price = info.get('market_price', 0.0)
        grid_net = info.get('total_import', 0) - info.get('total_export', 0)
        carbon = info.get('total_carbon_mass', 0)
        
        # Access SoC
        socs = [n.soc for n in vec_env.envs[0].nodes]
        
        # Store
        history['soc'].append(socs)
        history['price'].append(price)
        history['grid'].append(grid_net)
        history['carbon'].append(carbon)
        
        step_log = f"Hour {t}: Price=${price:.3f}/kWh | Grid={grid_net:.1f}kW | Carbon={carbon:.1f}g"
        logs.append(step_log)
        
        yield t, price, grid_net, carbon, logs, history
        
        time.sleep(0.1) # UI Pacing

# --------------------------------------------------------------------------
# 5. Execution Handler
# --------------------------------------------------------------------------
if run_btn:
    with st.spinner("Initializing Environment..."):
        sim_generator = run_simulation(
            grid_capacity_target, 
            use_ppo=(policy_mode == "PPO (Trained)"),
            deterministic=demo_mode
        )
        
    hist_soc = []
    hist_price = []
    
    for t, price, grid, carbon, log_lines, history in sim_generator:
        # Update Metrics
        m_hour.metric("Hour", f"{t}:00")
        m_price.metric("Market Price", f"${price:.3f}", delta_color="inverse")
        m_grid.metric("Grid Net Power", f"{grid:.1f} kW", delta=None)
        m_carbon.metric("Carbon Emission", f"{carbon:.1f} g")
        
        # Update Log
        log_text = "\n".join(reversed(log_lines[-5:])) # Show last 5
        log_area.code(log_text, language="text")
        
        # Live Plots
        # SoC - Plot top 10 Agents
        fig_soc, ax_soc = plt.subplots(figsize=(5,3))
        soc_array = np.array(history['soc']) / 50.0 * 100 # Approx Cap
        ax_soc.plot(soc_array) # Plot all lines so far
        ax_soc.set_title("SoC Trajectories (125 Agents)")
        ax_soc.set_ylabel("SoC %")
        ax_soc.set_ylim(0, 100)
        plot_spot_soc.pyplot(fig_soc)
        plt.close(fig_soc)
        
        # Price
        fig_price, ax_price = plt.subplots(figsize=(5,3))
        ax_price.plot(history['price'], color='green', marker='o')
        ax_price.set_title("Market Clearing Price")
        ax_price.set_ylabel("$/kWh")
        ax_price.set_ylim(0.0, 0.20)
        plot_spot_price.pyplot(fig_price)
        plt.close(fig_price)

    st.success("Simulation Complete!")
    
    # Save Results
    os.makedirs("results/demo_run", exist_ok=True)
    with open("results/demo_run/summary.txt", "w") as f:
        f.write(f"Total Carbon: {sum(history['carbon']):.2f} g\n")
        f.write(f"Peak Grid: {max(np.abs(history['grid'])):.2f} kW\n")
    st.toast("Results saved to results/demo_run/")
