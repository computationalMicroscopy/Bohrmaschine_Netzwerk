import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. INDUSTRIAL DASHBOARD SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Twin v20.1", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-container { 
        background-color: #161b22; border-left: 5px solid #58a6ff; 
        border-radius: 8px; padding: 15px; margin: 5px;
    }
    .main-cycle { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #79c0ff; font-weight: 800; }
    .sub-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .log-terminal { font-family: 'Consolas', monospace; font-size: 0.8rem; height: 400px; overflow-y: auto; background: #010409; padding: 15px; border: 1px solid #30363d; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MATERIAL-DATENBANK (INDUSTRIE-WERTE) ---
MATERIALIEN = {
    "Alu (G-AlSi10Mg)": {"kc1.1": 700, "mc": 0.2, "wear_rate": 0.01, "temp_crit": 150},
    "Stahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan (TiAl6V4)": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel 718": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 900}
}

# --- 3. KI-KERN (BAYESIAN INFERENCE) ---
@st.cache_resource
def get_inference_engine(n_v, n_t):
    model = DiscreteBayesianNetwork([
        ('ToolAge', 'State'), ('MechLoad', 'State'), ('ThermalStress', 'State'), ('Coolant', 'State'),
        ('State', 'Amplitude'), ('State', 'TempSens'), ('State', 'TorqueSens')
    ])
    cpd_age = TabularCPD('ToolAge', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('MechLoad', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('ThermalStress', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Coolant', 2, [[0.98], [0.02]])
    
    z_matrix = []
    for a in range(3):
        for l in range(2):
            for t in range(2):
                for c in range(2):
                    score = (a * 2) + (l * 4) + (t * 5) + (c * 7)
                    p2 = min(0.99, (score**2.4) / 300.0)
                    p1 = min(1.0-p2, score / 15.0)
                    z_matrix.append([1.0-p1-p2, p1, p2])
    
    cpd_state = TabularCPD('State', 3, np.array(z_matrix).T, 
                           evidence=['ToolAge', 'MechLoad', 'ThermalStress', 'Coolant'], 
                           evidence_card=[3, 2, 2, 2])
    
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state,
                   TabularCPD('Amplitude', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
                   TabularCPD('TempSens', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3]),
                   TabularCPD('TorqueSens', 2, [[0.99, 0.5, 0.01], [0.01, 0.5, 0.99]], ['State'], [3]))
    return VariableElimination(model)

# --- 4. SESSION MANAGEMENT ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'seed': np.random.RandomState(42)
    }

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🔩 Twin Control")
    mat_name = st.selectbox("Werkstoff wählen", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    
    with st.expander("Prozessdaten (CAM)", expanded=True):
        vc = st.slider("vc - Geschwindigkeit [m/min]", 20, 500, 160)
        f = st.slider("f - Vorschub [mm/U]", 0.02, 1.0, 0.18)
        d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
        cooling = st.toggle("Kühlschmierung", value=True)
    
    with st.expander("Maschinendynamik"):
        speed_idx = st.select_slider("Sim-Frequenz", options=[1000, 500, 100, 50, 10, 0], value=100)
        v_noise = st.slider("Vibrationsrauschen", 0.0, 1.0, 0.1)
        instability = st.slider("Aufspann-Starrheit", 0.0, 1.0, 0.05)

# --- 6. PHYSICS ENGINE ---
engine = get_inference_engine(v_noise, 0.05)

if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Drehmoment nach Kienzle
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc = (fc * d) / 2000
    
    # Verschleiß nach Taylor
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 600)
    s['wear'] += wear_inc * (1.5 if s['cycle'] > 500 else 1.0)
    
    # Thermik
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    
    # Amplitude (Vibration)
    amp = (0.005 + (s['wear'] * 0.003) + (instability * 0.2)) * (1 + s['seed'].normal(0, 0.1))
    
    # KI-Risikoanalyse
    evidence = {
        'ToolAge': 0 if s['cycle'] < 200 else (1 if s['cycle'] < 600 else 2),
        'MechLoad': 1 if mc > (d * 2.2) else 0,
        'ThermalStress': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Coolant': 0 if cooling else 1
    }
    risk = engine.query(['State'], evidence=evidence).values[2]
    
    if risk > 0.98 or s['wear'] > 150:
        s['broken'] = True
        s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc})
    s['logs'].insert(0, f"CYC {s['cycle']:04d} | Md: {mc:.2f}Nm | Risk: {risk:.2%}")

# --- 7. MAIN UI ---
st.title("🔩 Industrial Digital Twin: Drilling Analytics 20.1")

c_main, c_risk = st.columns([1, 2])
with c_main:
    st.markdown(f'<div class="metric-container"><span class="sub-label">Aktueller Zyklus</span><br><div class="main-cycle">{st.session_state.twin["cycle"]}</div></div>', unsafe_allow_html=True)

with c_risk:
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig_r = go.Figure()
        current_risk = df['r'].iloc[-1]
        line_color = '#3fb950' if current_risk < 0.5 else ('#d29922' if current_risk < 0.8 else '#f85149')
        
        fig_r.add_trace(go.Scatter(
            x=df['c'], y=df['r']*100, fill='tozeroy',
            fillcolor=f'rgba({248 if current_risk > 0.8 else 63}, {81 if current_risk > 0.8 else 185}, {73 if current_risk > 0.8 else 80}, 0.3)',
            line=dict(color=line_color, width=3), name="Bruch-Risiko"
        ))
        fig_r.update_layout(height=200, template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_r, use_container_width=True)

# Sensoren-Dashboard
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.twin['history'][-1] if st.session_state.twin['history'] else {'w':0,'amp':0,'t':22,'mc':0}
m1.metric("Vibration (mm)", f"{last['amp']:.4f}")
m2.metric("Drehmoment (Nm)", f"{last['mc']:.2f}")
m3.metric("Temperatur (°C)", f"{last['t']:.1f}")
m4.metric("Verschleiß (%)", f"{last['w']:.1f}")

st.divider()

g_left, g_right = st.columns([2, 1])
with g_left:
    if st.session_state.twin['history']:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Drehmoment (Nm)", line=dict(color='#58a6ff')))
        fig.add_trace(go.Scatter(x=df['c'], y=df['t'], name="Temperatur (°C)", line=dict(color='#f85149')), secondary_y=True)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with g_right:
    if st.button("▶️ START / STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
    if st.button("🔄 SYSTEM-RESET", use_container_width=True):
        st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
        st.rerun()
    if st.session_state.twin['broken']: st.error("WERKZEUGBRUCH DETEKTIERT!")
    log_content = "".join([f"<div>{l}</div>" for l in st.session_state.twin['logs'][:50]])
    st.markdown(f'<div class="log-terminal">{log_content}</div>', unsafe_allow_html=True)

if st.session_state.twin['active']:
    if speed_idx > 0: time.sleep(speed_idx/1000)
    st.rerun()
