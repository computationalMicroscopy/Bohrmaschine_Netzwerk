import streamlit as st
import pandas as pd
import numpy as np
import time

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. SETUP & DESIGN ---
st.set_page_config(layout="wide", page_title="KI-Bohrsystem v3.5 Pro", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 15px; margin-bottom: 10px;
    }
    .sandbox-card {
        background: rgba(63, 185, 80, 0.08);
        border: 1px solid #3fb950; border-radius: 12px; padding: 20px;
    }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; }
    .blue-glow { color: #58a6ff; text-shadow: 0 0 10px rgba(88, 166, 255, 0.4); }
    .red-glow { color: #f85149; text-shadow: 0 0 10px rgba(248, 81, 73, 0.4); }
    .xai-bar-bg { background: #30363d; border-radius: 4px; height: 6px; width: 100%; margin: 4px 0 10px 0; }
    .xai-bar-fill { height: 6px; border-radius: 4px; transition: width 0.4s; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-ENGINE ---
@st.cache_resource
def get_engine():
    model = BayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State')])
    # CPD Definitionen (Bayesian Logic)
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.95], [0.05]])
    z_matrix = []
    for age in range(3):
        for load in range(2):
            for therm in range(2):
                for cool in range(2):
                    score = (age * 2) + (load * 4) + (therm * 7) + (cool * 8)
                    v = [0.99, 0.005, 0.005] if score <= 3 else ([0.6, 0.35, 0.05] if score <= 8 else ([0.15, 0.45, 0.4] if score <= 12 else [0.01, 0.04, 0.95]))
                    z_matrix.append(v)
    cpd_state = TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2])
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state)
    return VariableElimination(model)

def categorize(cycle, wear, temp, md, mat_crit, cooling_active, sens_load):
    a_c = 0 if cycle < 250 else (1 if cycle < 650 else 2)
    l_c = 1 if md > ((12.0 * 2.2) / sens_load) else 0
    t_c = 1 if temp > mat_crit else 0
    c_c = 0 if cooling_active else 1
    return a_c, l_c, t_c, c_c

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'active': False, 'broken': False,
                             't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🎮 Live-Steuerung")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 500, 160)
    f = st.slider("f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Bohrer-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung", value=True)
    st.divider()
    sens_vib = st.slider("Vibrations-Gain", 0.1, 5.0, 1.0)
    sim_speed = st.select_slider("Abtastrate (ms)", options=[200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK (Digitaler Zwilling) ---
engine = get_engine()
s = st.session_state.twin

if s['active'] and not s['broken']:
    s['cycle'] += 1
    # Physik-Simulation
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    md_val = (fc * d) / 2000
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300))
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.5)
    
    # Sensorik (Vibrationen)
    vib = (((0.01 + (s['wear'] * 0.005)) * 10) + s['seed'].normal(0, 0.05)) * sens_vib
    
    # KI-Inferenz
    a, l, t, c = categorize(s['cycle'], s['wear'], s['t_current'], md_val, mat['temp_crit'], cooling, 1.0)
    s['risk'] = engine.query(['State'], evidence={'Age': a, 'Load': l, 'Therm': t, 'Cool': c}).values[2]
    
    if s['risk'] > 0.98 or s['wear'] > 100: s['broken'] = True; s['active'] = False
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'md': md_val, 'v': vib})

# --- 6. UI LAYOUT ---
st.title("🗜️ PROZESS-MONITOR | Digitaler Zwilling v3.5")

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("### 📊 Sensor-Daten")
    st.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main {"red-glow" if s["risk"]>0.5 else "blue-glow"}">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{s["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{s["wear"]:.1f} %</span></div>', unsafe_allow_html=True)

with col_mid:
    st.markdown("### 📈 Live-Oszilloskop")
    if len(s['history']) > 0:
        df = pd.DataFrame(s['history']).suffix_label = "Zyklen"
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Risiko & Verschleiß", "Sensorik: Vibration & Moment"))
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="Risiko %", fill='tozeroy', line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['w'], name="Verschleiß %", line=dict(color='#e3b341')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['v'], name="Vibration [g]", line=dict(color='#bc8cff')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['md'], name="Moment [Nm]", line=dict(color='#58a6ff')), row=2, col=1)
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Simulation starten, um Sensor-Feeds zu empfangen...")

with col_right:
    st.markdown("### 🧪 Strategie-Planer")
    st.markdown('<div class="sandbox-card">', unsafe_allow_html=True)
    h_vc = st.slider("Plan: vc", 20, 500, vc)
    h_cool = st.toggle("Plan: Kühlung", value=cooling)
    
    # What-If Berechnung
    h_md = (mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2) * d) / 2000
    h_t = 22 + (s['wear'] * 1.5) + (h_vc * 0.2) + (0 if h_cool else 250)
    ha, hl, ht, hc = categorize(s['cycle'], s['wear'], h_t, h_md, mat['temp_crit'], h_cool, 1.0)
    h_risk = engine.query(['State'], evidence={'Age': ha, 'Load': hl, 'Therm': ht, 'Cool': hc}).values[2]
    
    st.metric("Projiziertes Risiko", f"{h_risk:.1%}", delta=f"{h_risk - s['risk']:.1%}", delta_color="inverse")
    
    st.markdown("**Ursachen-Analyse (XAI):**")
    for lbl, v, clr in [("Alter", ha/2, "#58a6ff"), ("Last", hl, "#e3b341"), ("Hitze", ht, "#f85149")]:
        st.markdown(f'<div style="font-size:11px">{lbl}</div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{v*100}%; background:{clr}"></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
c1, c2, c3 = st.columns([1,1,2])
if c1.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
if c2.button("🔄 RESET", use_container_width=True):
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}
    st.rerun()
if s['broken']: st.error("🚨 WERKZEUGBRUCH DETEKTIERT! System gestoppt.")

if s['active']:
    time.sleep(sim_speed / 1000); st.rerun()
