
import streamlit as st
import json
import pandas as pd
import time
import os
import plotly.express as px

st.set_page_config(page_title="SLIM Real-Time Energy Market", layout="wide")

st.title("⚡ SLIM v3: Scientific P2P Market Monitor")
st.markdown("Monitoring live energy transactions with scientific integrity. Metrics reflect corrected supply-demand economics.")

STATS_FILE = "live_market_stats.json"

def load_data():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return None
    return None

placeholder = st.empty()

# Interactive Controls (Outside Loop)
st.sidebar.title("Controls")
show_raw = st.sidebar.checkbox("Show Raw Live Data", key="show_raw_data")
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 0.5, 5.0, 1.0)

while True:
    data = load_data()
    
    with placeholder.container():
        if data:
            curr_step = data['history'][-1]['step'] if data['history'] else 0
            
            # Row 1: Economic Performance
            st.header("1. Economic Performance")
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric("Market Profit", f"${data['cumulative_market_profit']:.2f}", 
                      delta=f"${data['history'][-1]['profit']:.2f} (Instant)")
            
            # Economic Profit (Market - Degradation)
            m2.metric("Economic Profit", f"${data['cumulative_economic_profit']:.2f}",
                      help="Net profit after accounting for battery degradation and wear.")
            
            m3.metric("Total P2P Volume", f"{data['total_p2p_volume']:.2f} kWh")
            
            m4.metric("Last Update", data['last_update'])

            st.divider()

            # Row 2: Physical Dependency & Carbon
            st.header("2. Scientific Metrics")
            c1, c2, c3 = st.columns([2, 1, 1])
            
            with c1:
                st.subheader("Grid Dependency (Rolling 50-Step)")
                # Gauge-like area chart for dependency
                history_df = pd.DataFrame(data['history'])
                chart_key_g = f"grid_dep_chart_{curr_step}_{time.time()}"
                fig_g = px.area(history_df, x='step', y='grid_dep', 
                                template="plotly_dark", 
                                color_discrete_sequence=['#EF553B'],
                                range_y=[0, 100])
                st.plotly_chart(fig_g, use_container_width=True, key=chart_key_g)
                
            with c2:
                st.subheader("Lifetime Stats")
                st.write(f"**Cumulative Dependency:** {data['cumulative_grid_dependency']:.1f}%")
                # Carbon Reduction calculation (Relative to baseline)
                reduction = 0
                if data['cumulative_baseline_co2'] > 0:
                    reduction = (data['cumulative_baseline_co2'] - data['cumulative_co2']) / data['cumulative_baseline_co2'] * 100
                st.write(f"**CO₂ Reduction:** {reduction:.1f}%")
                st.progress(max(0, min(100, int(reduction))), text="Reduction Progress")

            with c3:
                st.subheader("Market Quality")
                st.write(f"**Trade Success Rate:** {data['trade_success_rate']:.1%}")
                st.write(f"**Rolling Profit (50s):** ${data['rolling_market_profit']:.2f}")

            # History Chart
            st.divider()
            st.subheader("P2P Volume Stream")
            chart_key_v = f"p2p_vol_chart_{curr_step}_{time.time()}"
            fig_v = px.line(history_df, x='step', y='p2p_v', 
                            template="plotly_dark", 
                            color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_v, use_container_width=True, key=chart_key_v)

            # Raw Data View
            if show_raw:
                st.divider()
                st.subheader("Raw Live Data Stream")
                st.json(data)
        else:
            st.warning("Waiting for live simulation data... Ensure 'real_time_loop.py' is running.")
            
    time.sleep(refresh_rate)
