
import streamlit as st
import pandas as pd
import numpy as np
import sys
# --- Monkey Patch for NumPy 1.x vs 2.0 Mismatch ---
try:
    import numpy.core
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = numpy.core
    if 'numpy._core.numeric' not in sys.modules:
        from numpy.core import numeric
        sys.modules['numpy._core.numeric'] = numeric
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except ImportError:
    pass

import numpy.random 
try:
    from numpy.random import PCG64
except ImportError:
    pass

import time
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train.energy_env_robust import EnergyMarketEnvRobust
except ImportError:
    st.error("Could not import EnergyMarketEnvRobust. Make sure you are running from the project root.")

# Page Config
st.set_page_config(page_title="P2P Energy Trading Demo", layout="wide")

st.title("⚡ P2P Energy Trading: Autonomous Market Demo")
st.markdown("### Reinforcement Learning Implementation")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
data_file = st.sidebar.text_input("Data File", "test_day_profile.csv")
model_path = st.sidebar.text_input("Model Path", "models_demo/ppo_energy_final.zip")
random_day = st.sidebar.checkbox("Random Start Day", value=True)
fps = st.sidebar.slider("Simulation Speed (FPS)", 1, 10, 4)

# --- Session State ---
if 'env' not in st.session_state:
    st.session_state['env'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'obs' not in st.session_state:
    st.session_state['obs'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'running' not in st.session_state:
    st.session_state['running'] = False

# --- Initialization Function ---
def init_simulation():
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        return
    if not os.path.exists(data_file):
        st.error(f"Data file not found: {data_file}")
        return

    # Create Env
    def make_env():
        return EnergyMarketEnvRobust(
            n_agents=4, 
            data_file=data_file,
            random_start_day=random_day, 
            forecast_horizon=1
        )
    
    st.session_state['env'] = DummyVecEnv([make_env])
    
    # LOAD VECNORMALIZE - DISABLED FOR STABILITY
    # model_dir = Path(model_path).parent
    # vecnorm_path = model_dir / "vec_normalize.pkl"
    # if vecnorm_path.exists():
    #     try:
    #         st.session_state['env'] = VecNormalize.load(str(vecnorm_path), st.session_state['env'])
    #         st.session_state['env'].training = False
    #         st.session_state['env'].norm_reward = False
    #         st.sidebar.success("Loaded Normalization Stats")
    #     except Exception as e:
    #         st.error(f"VecNormalize Load Error (Ignored): {e}")

    # Load Model
    st.session_state['model'] = PPO.load(model_path, env=st.session_state['env'])
    st.session_state['obs'] = st.session_state['env'].reset()
    st.session_state['history'] = []
    st.sidebar.success("Model & Environment loaded!")

# --- Buttons ---
col1, col2 = st.sidebar.columns(2)
if col1.button("Initialize / Reset"):
    init_simulation()
    
if col2.button("Run / Pause"):
    st.session_state['running'] = not st.session_state['running']

# --- Main Loop ---
placeholder = st.empty()

try:
    if st.session_state['env'] is not None:
        while st.session_state['running']:
            # Step
            action, _ = st.session_state['model'].predict(st.session_state['obs'], deterministic=True)
            obs, rewards, dones, infos = st.session_state['env'].step(action)
            st.session_state['obs'] = obs
            
            info = infos[0]
            
            # Access Inner Env for Details
            if isinstance(st.session_state['env'], VecNormalize):
                inner_env = st.session_state['env'].venv.envs[0]
            else:
                inner_env = st.session_state['env'].envs[0]
                
            step_idx = inner_env.timestep_count
            
            # Data Collection
            snapshot = {
                "step": step_idx,
                "market_price": info.get("market_price", 0.0),
                "grid_flow": info.get("total_export", 0) - info.get("total_import", 0),
                "reward": rewards[0],
                "gini": info.get("gini_index", 0.0)
            }
            
            # Agent Details
            nodes = inner_env.nodes
            for i, node in enumerate(nodes):
                snapshot[f"agent_{i}_soc"] = node.soc
                
            st.session_state['history'].append(snapshot)
            
            # --- Visualization Code ---
            with placeholder.container():
                df = pd.DataFrame(st.session_state['history'])
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Step", step_idx)
                m2.metric("Market Price ($/kWh)", f"{snapshot['market_price']:.3f}")
                m3.metric("Grid Flow (kW)", f"{snapshot['grid_flow']:.2f}", delta_color="normal")
                m4.metric("Gini (Inequality)", f"{snapshot['gini']:.3f}")
                
                # Charts
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Battery State of Charge (kWh)")
                    if not df.empty:
                        soc_cols = [c for c in df.columns if "_soc" in c]
                        fig_soc = px.line(df, x="step", y=soc_cols, height=300)
                        fig_soc.update_layout(yaxis_range=[0, 50]) # Assuming 50kWh cap
                        st.plotly_chart(fig_soc, use_container_width=True)
                
                with c2:
                    st.subheader("Market Price & Grid Interaction")
                    if not df.empty:
                         # Create dual-axis chart
                        fig_market = go.Figure()
                        fig_market.add_trace(go.Scatter(x=df['step'], y=df['market_price'], name="Price ($)", line=dict(color='green')))
                        fig_market.add_trace(go.Scatter(x=df['step'], y=df['grid_flow'], name="Grid Flow (kW)", yaxis="y2", line=dict(color='blue', dash='dot')))
                        
                        fig_market.update_layout(
                            yaxis=dict(title="Price ($)"),
                            yaxis2=dict(title="Grid Flow (kW)", overlaying="y", side="right"),
                            height=300,
                            legend=dict(x=0, y=1.1, orientation="h")
                        )
                        st.plotly_chart(fig_market, use_container_width=True)
                
            time.sleep(1.0 / fps)
            
            if dones[0]:
                st.session_state['running'] = False
                st.info("Episode Completed.")
                break

    else:
        st.info("👈 Click 'Initialize / Reset' to start the demo.")
        st.markdown("""
        ### How to use this Demo:
        1.  **Install Requirements**: `pip install streamlit plotly`
        2.  **Initialize**: Load the model and environment.
        3.  **Run**: Watch the agents trade in real-time.
        
        ### Key Features:
        *   **Double Auction**: See how prices fluctuate with supply/demand.
        *   **Battery Ops**: Watch Agents charge/discharge to arbitrage prices.
        *   **Zero-Carbon**: Agents prefer P2P over dirty grid power.
        """)
except Exception as e:
    st.error(f"An unexpected runtime error occurred: {e}")
    st.exception(e)
