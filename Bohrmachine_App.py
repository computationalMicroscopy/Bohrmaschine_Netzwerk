import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & INDUSTRIAL THEME ---
st.set_page_config(layout="wide", page_title="AI Twin v20.2 Trainer", page_icon="🎓")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-container { 
        background-color: #161b22; border-left: 5px solid #58a6ff; 
        border-radius: 8px; padding: 15px; margin: 5px;
    }
    .main-cycle { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #79c0ff; font-weight: 800; }
    .sub-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .log-terminal { font-family: 'Consolas', monospace; font-size: 0.8rem; height: 300px; overflow-y: auto; background: #010409; padding: 15px; border: 1px solid #30363d; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION MANAGEMENT & GLOBAL STATE ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'sabotage_coolant': False, 'material_anomaly': 1.0, 'seed': np.random.RandomState(42)
    }

# --- 3. TEACHER PANEL (SABOTAGE TOOLS FOR KI-TRAINING) ---
with st.expander("🛠️ TEACHER PANEL (Hidden Control)", expanded=False):
    st.warning("Nutze diese Regler, um die Reaktion der KI und der Schüler zu testen.")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        sab_cool = st.toggle("Kühlsystem manipulieren (Leckage simulieren)", value=False)
        st.session_state.twin['sabotage_coolant'] = sab_cool
    with col_t2:
        anom = st.slider("Material-Anomalie (Lokaler Härteeinschluss)", 1.0, 3.0, 1.0)
        st.session_state.twin['material_anomaly'] = anom

# --- 4. MATERIAL-DATENBANK & PHYSIK ---
MATERIALIEN = {
    "Stahl 42CrMo4": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Titan TiAl6V4": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel 718": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

with st.sidebar:
    st.title("🔩 Twin Settings")
    mat_name = st.selectbox("Werkstoff wählen", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    
    with st.expander("Prozessdaten (CAM)", expanded=True):
        vc = st.slider("vc - Geschwindigkeit [m/min]", 20, 500, 160)
        f = st.slider("f - Vorschub [mm/U]", 0.02, 1.0, 0.18)
        d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
        cooling_ui = st.toggle("Kühlschmierung aktiv", value=True)
    
    with st.expander("Maschinendynamik"):
        sim_speed = st.select_slider("Sim-Frequenz", options=[500, 100, 50, 10, 0], value=50)
        v_noise = st.slider("Vibrationsrauschen", 0.0, 1.0, 0.1)

# --- 5. BAYESIAN NETWORK ARCHITECTURE ---
@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([
        ('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'),
        ('State', 'Amp'), ('State', 'Temp')
    ])
    
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.98], [0.02]])
    
    z_matrix = []
    for a in range(3):
        for l in range(2):
            for t in range(2):
                for c in range(2):
                    score = (a * 2) + (l * 4) + (t * 5) + (c * 7)
                    p2 = min(0.99, (score**2.5) / 350.0)
                    p1 = min(1.0-p2, score / 15.0)
                    z_matrix.append([1.0-p1-p2, p1, p2])
    
    model.add_cpds(
        cpd_age, cpd_load, cpd_therm, cpd_cool,
        TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2]),
        TabularCPD('Amp', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
        TabularCPD('Temp', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3])
    )
    return VariableElimination(model)

# --- 6. PHYSICS & INFERENCE LOOP ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Berücksichtigung der Sabotage-Aktionen
    eff_cooling = cooling_ui and not s['sabotage_coolant']
    eff_load_factor = s['material_anomaly']
    
    # 6a. Physikalische Berechnung (Kienzle & Energiebilanz)
    fc = mat['kc1.1'] * eff_load_factor * (f** (1-mat['mc'])) * (d/2)
    mc = (fc * d) / 2000
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if eff_cooling else 400)
    s['wear'] += wear_inc * (1.5 if s['cycle'] > 500 else 1.0)
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if eff_cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    amp = (0.005 + (s['wear'] * 0.003)) * (1 + s['seed'].normal(0, 0.1))
    
    # 6b. KI-Inferenz
    engine = get_engine()
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 200 else (1 if s['cycle'] < 600 else 2),
        'Load': 1 if mc > (d * 2.2) else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if eff_cooling else 1
    }).values[2]
    
    # 6c. Ausfall-Logik
    if risk > 0.98 or s['wear'] > 150:
        s['broken'] = True
        s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc})
    if s['sabotage_coolant']: s['logs'].insert(0, f"CYC {s['cycle']:04d} | ⚠️ THERMAL ALERT: COOLANT LOSS")
    s['logs'].insert(0, f"CYC {s['cycle']:04d} | Md: {mc:.2f}Nm | Risk: {risk:.1%}")

# --- 7. UI DASHBOARD ---
st.title("🔩 AI Precision Twin v20.2: Trainer Edition")

c_stat, c_graph = st.columns([1, 2])
with c_stat:
    st.markdown(f'<div class="metric-container"><span class="sub-label">Kumulative Zyklen</span><br><div class="main-cycle">{st.session_state.twin["cycle"]}</div></div>', unsafe_allow_html=True)
    if st.button("▶️ PROZESS START/STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
    if st.button("🔄 SYSTEM-RESET", use_container_width=True):
        st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'sabotage_coolant':False,'material_anomaly':1.0,'seed':np.random.RandomState(42)}
        st.rerun()

with c_graph:
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="Bruch-Risiko %", line=dict(color='#f85149', width=3)))
        fig_r.update_layout(height=220, template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0,105]))
        st.plotly_chart(fig_r, use_container_width=True)

# Sensoren
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.twin['history'][-1] if st.session_state.twin['history'] else {'w':0,'amp':0,'t':22,'mc':0}
with m1: st.markdown(f'<div class="metric-container"><span class="sub-label">Vibrations-Amp.</span><br><span style="font-size:1.5rem; font-weight:bold; color:#58a6ff">{last["amp"]:.4f} mm</span></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-container"><span class="sub-label">Drehmoment</span><br><span style="font-size:1.5rem; font-weight:bold; color:#58a6ff">{last["mc"]:.2f} Nm</span></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-container"><span class="sub-label">Temperatur</span><br><span style="font-size:1.5rem; font-weight:bold; color:#f85149">{last["t"]:.1f} °C</span></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="metric-container"><span class="sub-label">Verschleiß</span><br><span style="font-size:1.5rem; font-weight:bold; color:#e3b341">{last["w"]:.1f} %</span></div>', unsafe_allow_html=True)

st.divider()

g_left, g_right = st.columns([2, 1])
with g_left:
    if st.session_state.twin['history']:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Drehmoment (Nm)", line=dict(color='#58a6ff')))
        fig.add_trace(go.Scatter(x=df['c'], y=df['t'], name="Temp (°C)", line=dict(color='#f85149')), secondary_y=True)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with g_right:
    if st.session_state.twin['broken']: st.error("⚠️ WERZEUGBRUCH: STOPP ERZWUNGEN")
    log_content = "".join([f"<div style='border-bottom:1px solid #21262d; padding:2px;'>{l}</div>" for l in st.session_state.twin['logs'][:50]])
    st.markdown(f'<div class="log-terminal">{log_content}</div>', unsafe_allow_html=True)

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
