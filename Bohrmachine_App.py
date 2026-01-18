import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- SETUP ---
st.set_page_config(layout="wide", page_title="AI Twin v20.2 Trainer", page_icon="🎓")

# --- INITIAL STATE ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'sabotage_coolant': False, 'material_anomaly': 1.0, 'seed': np.random.RandomState(42)
    }

# --- TEACHER PANEL (SABOTAGE TOOLS) ---
with st.expander("🛠️ TEACHER PANEL (Hidden Control)", expanded=False):
    st.warning("Nutze diese Regler, um die Reaktion der KI und der Schüler zu testen.")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        sab_cool = st.toggle("Kühlsystem manipulieren (Leckage)", value=False)
        st.session_state.twin['sabotage_coolant'] = sab_cool
    with col_t2:
        anom = st.slider("Material-Anomalie (Härteeinschluss)", 1.0, 3.0, 1.0)
        st.session_state.twin['material_anomaly'] = anom

# --- SIDEBAR & PHYSIK-DATABASE ---
MATERIALIEN = {
    "Stahl 42CrMo4": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Titan TiAl6V4": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel 718": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

with st.sidebar:
    st.title("🔩 Twin Settings")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 500, 160)
    f = st.slider("f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Ø [mm]", 1.0, 60.0, 12.0)
    cooling_ui = st.toggle("Kühlschmierung aktiv", value=True)
    sim_speed = st.select_slider("Sim-Speed", options=[500, 100, 50, 10, 0], value=50)

# --- BAYESIAN ENGINE ---
@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([
        ('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'),
        ('State', 'Amp'), ('State', 'Temp')
    ])
    # Definition der CPTs (Logik wie in Doku beschrieben)
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.98], [0.02]])
    
    # CPT State (Wahrheitstabelle für 36 Kombinationen)
    z_matrix = []
    for a in range(3):
        for l in range(2):
            for t in range(2):
                for c in range(2):
                    score = (a * 2) + (l * 4) + (t * 5) + (c * 7)
                    p2 = min(0.99, (score**2.5) / 350.0)
                    p1 = min(1.0-p2, score / 15.0)
                    z_matrix.append([1.0-p1-p2, p1, p2])
    
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool,
                   TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2]),
                   TabularCPD('Amp', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
                   TabularCPD('Temp', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3]))
    return VariableElimination(model)

# --- SIMULATION LOOP ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Sabotage-Check
    eff_cooling = cooling_ui and not s['sabotage_coolant']
    eff_load_factor = s['material_anomaly']
    
    # Physik
    mc = (mat['kc1.1'] * eff_load_factor * (f** (1-mat['mc'])) * (d/2) * d) / 2000
    s['wear'] += (mat['wear_rate'] * (vc**1.6) * f) / (15000 if eff_cooling else 400)
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if eff_cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    amp = (0.005 + (s['wear'] * 0.003)) * (1 + s['seed'].normal(0, 0.1))
    
    # KI-Inferenz
    engine = get_engine()
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 200 else (1 if s['cycle'] < 600 else 2),
        'Load': 1 if mc > (d * 2.2) else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if eff_cooling else 1
    }).values[2]
    
    if risk > 0.98 or s['wear'] > 150: s['broken'] = True; s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc})
    if s['sabotage_coolant']: s['logs'].insert(0, "⚠️ SENSOR ALERT: COOLANT PRESSURE DROP DETECTED")

# --- UI RENDERING ---
st.title("🔩 AI Twin v20.2: Trainer Edition")

# Top Stats
c_top1, c_top2 = st.columns([1, 2])
with c_top1:
    st.metric("Bohr-Zyklus", st.session_state.twin['cycle'])
    if st.button("▶️ START/STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'sabotage_coolant':False,'material_anomaly':1.0,'seed':np.random.RandomState(42)}
        st.rerun()

with c_top2:
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig_r = go.Figure(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', line=dict(color='#ff4b4b')))
        fig_r.update_layout(height=200, template="plotly_dark", title="KI-Bruchrisiko %", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_r, use_container_width=True)

# Main Telemetry
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.twin['history'][-1] if st.session_state.twin['history'] else {'w':0,'amp':0,'t':22,'mc':0}
m1.metric("Amplitude", f"{last['amp']:.4f} mm")
m2.metric("Drehmoment", f"{last['mc']:.2f} Nm")
m3.metric("Temperatur", f"{last['t']:.1f} °C")
m4.metric("Verschleiß", f"{last['w']:.1f} %")

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
