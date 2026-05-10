import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.inference import MTUEnginePredictor
from agent.graph import build_graph
from agent.tools import SENSOR_MAP
from utils.voice_query import handle_voice_query

st.set_page_config(layout="wide", page_title="MTU SensorMind Co-Pilot", page_icon="✈️")

# MRO Mock Database
MRO_HISTORY = {
    1: "Overhauled 200 cycles ago. HPC blades replaced.",
    2: "Fuel metering unit replaced 15 cycles ago.",
    3: "Routine A-Check 50 cycles ago. No anomalies noted.",
    4: "Variable Stator Vane (VSV) actuator calibrated recently.",
    5: "Fan blades inspected for erosion; minor wear detected."
}

@st.cache_resource
def load_ml_predictor():
    predictor = MTUEnginePredictor()
    try:
        predictor.load_models()
        return predictor
    except Exception as e:
        return None

@st.cache_resource
def load_agent_graph():
    return build_graph()

@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train_FD001.txt')
    if not os.path.exists(data_path):
        return None
    COLUMNS = ['engine_id', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(data_path, sep='\s+', header=None, names=COLUMNS)
    return df

if 'selected_engine' not in st.session_state: st.session_state.selected_engine = 1
if 'current_cycle' not in st.session_state: st.session_state.current_cycle = 30
if 'agent_trace' not in st.session_state: st.session_state.agent_trace = []
if 'generated_wos' not in st.session_state: st.session_state.generated_wos = []

predictor = load_ml_predictor()
agent_graph = load_agent_graph()
df = load_dataset()

st.markdown("""
<style>
    .critical {color: white; background-color: #ff4b4b; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .high {color: white; background-color: #ff8c00; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .medium {color: black; background-color: #ffe100; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .low {color: white; background-color: #00cc66; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .data_fault {color: white; background-color: #8b0000; padding: 5px 10px; border-radius: 5px; font-weight: bold;}
    .trace-box {background-color: #1e1e1e; color: #00ff00; padding: 15px; font-family: monospace; border-radius: 8px; height: 300px; overflow-y: scroll;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Rolls-Royce_logo.svg/1200px-Rolls-Royce_logo.svg.png", width=150)
    st.title("MTU SensorMind")
    st.caption("Agentic Equipment Health Co-Pilot")
    st.markdown("---")
    
    if df is None or predictor is None:
        st.error("⚠️ System Offline: Models or Data not found.")
        st.stop()
        
    engine_list = df['engine_id'].unique()
    st.session_state.selected_engine = st.selectbox("Select Engine ID", engine_list, index=int(np.where(engine_list == st.session_state.selected_engine)[0][0]))
    max_cycle = df[df['engine_id'] == st.session_state.selected_engine]['cycle'].max()
    st.session_state.current_cycle = st.slider("Simulation Cycle (Time)", min_value=30, max_value=int(max_cycle), value=min(st.session_state.current_cycle, int(max_cycle)))
    
    if st.button("Advance +5 Cycles"):
        st.session_state.current_cycle = min(st.session_state.current_cycle + 5, int(max_cycle))
        st.rerun()
        
    st.markdown("---")
    st.subheader("🎙️ Voice Command")
    if st.button("🎤 Record Query"):
        with st.spinner("Listening..."):
            res = handle_voice_query()
            if "error" in res: st.error(res["error"])
            else:
                st.success(f"Recognized: '{res['raw_text']}'")
                if res["engine_id"] and res["engine_id"] in engine_list: st.session_state.selected_engine = res["engine_id"]

engine_df = df[(df['engine_id'] == st.session_state.selected_engine) & (df['cycle'] <= st.session_state.current_cycle)]
latest_window = engine_df.tail(30)
service_hist = MRO_HISTORY.get(st.session_state.selected_engine, "Routine maintenance only. No major overhauls recently.")

try:
    status = predictor.get_engine_status(st.session_state.selected_engine, latest_window)
except Exception as e:
    st.error(f"Inference error: {e}")
    st.stop()

st.header(f"Engine {st.session_state.selected_engine} Status Dashboard")

# Data Quality Gate UI
if status.get('data_quality_warnings'):
    for warn in status['data_quality_warnings']:
        st.error(f"🔧 **{warn}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Cycle", st.session_state.current_cycle)
col2.metric("Anomaly Score", f"{status['anomaly_score']:.3f}", delta="High" if status['is_anomaly'] else "Normal", delta_color="inverse")
col3.metric("Predicted RUL", f"{status['rul_prediction']} cycles", delta="-1" if status['rul_prediction'] < 50 else "+", delta_color="normal")
risk_color = status['risk_level'].lower()
col4.markdown(f"**Risk Level:**<br><span class='{risk_color}'>{status['risk_level']}</span>", unsafe_allow_html=True)

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["Deep Dive", "Agent Co-Pilot", "What-If Simulator", "Work Orders"])

with tab1:
    st.info(f"📋 **MRO Service History:** {service_hist}")
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        st.subheader("Sensor Trends (Last 50 Cycles)")
        plot_df = engine_df.tail(50)
        top_sensors = sorted(status['sensor_deviations'].items(), key=lambda x: x[1], reverse=True)[:3]
        top_sensor_names = [s[0] for s in top_sensors]
        fig = go.Figure()
        for s in top_sensor_names:
            fig.add_trace(go.Scatter(x=plot_df['cycle'], y=plot_df[s], mode='lines+markers', name=SENSOR_MAP.get(s, s)))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_c2:
        st.subheader("RUL Gauge")
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = status['rul_prediction'],
            domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Cycles Remaining"},
            gauge = {'axis': {'range': [0, 150]}, 'bar': {'color': "darkblue"},
                     'steps': [{'range': [0, 30], 'color': "red"}, {'range': [30, 80], 'color': "orange"}, {'range': [80, 150], 'color': "lightgreen"}],
                     'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': status['rul_prediction']}}
        ))
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(gauge, use_container_width=True)

with tab2:
    st.subheader("Agentic Diagnostics & Planning (Guided Troubleshooting)")
    if st.button("🚀 Run Diagnostic Agent", type="primary"):
        st.session_state.agent_trace = []
        trace_placeholder = st.empty()
        
        initial_state = {
            "engine_id": status['engine_id'],
            "sensor_deviations": status['sensor_deviations'],
            "anomaly_score": status['anomaly_score'],
            "is_anomaly": status['is_anomaly'],
            "rul": status['rul_prediction'],
            "risk_level": status['risk_level'],
            "delay_cycles": 10,
            "service_history": service_hist
        }
        
        for output in agent_graph.stream(initial_state):
            for key, value in output.items():
                if 'trace_log' in value:
                    st.session_state.agent_trace = value['trace_log']
                    trace_html = "<div class='trace-box'>" + "<br>".join(st.session_state.agent_trace) + "</div>"
                    trace_placeholder.markdown(trace_html, unsafe_allow_html=True)
                    time.sleep(0.5)
                if 'work_order_path' in value:
                    wo_path = value['work_order_path']
                    if wo_path not in st.session_state.generated_wos:
                        st.session_state.generated_wos.append(wo_path)
    else:
        if st.session_state.agent_trace:
            trace_html = "<div class='trace-box'>" + "<br>".join(st.session_state.agent_trace) + "</div>"
            st.markdown(trace_html, unsafe_allow_html=True)
        else:
            st.info("Click 'Run Diagnostic Agent' to analyze the current engine state.")

with tab3:
    st.subheader("Maintenance Delay Simulator")
    delay = st.slider("Delay Maintenance by (Cycles)", min_value=1, max_value=50, value=10)
    if st.button("Simulate Degradation"):
        from agent.tools import simulate_whatif
        res = simulate_whatif(status['rul_prediction'], status['anomaly_score'], delay)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current RUL", res['current_rul'])
        c2.metric("Projected RUL", res['projected_rul_after_delay'], delta=res['projected_rul_after_delay'] - res['current_rul'])
        c3.metric("Failure Prob. Increase", f"+{res['failure_probability_increase']}%")
        if "CRITICAL" in res['recommendation']: st.error(res['recommendation'])
        elif "HIGH" in res['recommendation']: st.warning(res['recommendation'])
        else: st.success(res['recommendation'])

with tab4:
    st.subheader("Generated Work Orders")
    if not st.session_state.generated_wos:
        st.info("No work orders generated yet. Run the Diagnostic Agent.")
    else:
        for wo in reversed(st.session_state.generated_wos):
            filename = os.path.basename(wo)
            st.markdown(f"📄 **{filename}**")
            with open(wo, "rb") as file:
                btn = st.download_button(label=f"Download {filename}", data=file, file_name=filename, mime="application/pdf")
            st.markdown("---")
