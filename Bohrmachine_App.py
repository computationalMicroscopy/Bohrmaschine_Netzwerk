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
st.set_page_config(layout="wide", page_title="AI Precision Twin v20.3 Predictive", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .predictive-card { 
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d; border-left: 5px solid #e3b341;
        border-radius: 10px; padding: 20px; text-align: center;
    }
    .ttf-value { font-family: 'JetBrains Mono', monospace; font-size: 3rem; color: #e3b341; font-weight: bold; }
    .status-ok { color: #3fb950; font-weight: bold; }
    .status-warn { color: #d29922; font-weight: bold; }
    .status-crit { color: #f85149; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION MANAGEMENT ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'seed': np.random.RandomState(42)
    }

# --- 3. KI-KERN & PHYSIK-PARAMETER ---
MATERIALIEN = {
    "Stahl 42CrMo4": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Titan TiAl6V4": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel 718": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([
        ('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'),
        ('State', 'Amp'), ('State', 'Temp')
    ])
    # Definition der bedingten Wahrscheinlichkeiten (CPTs)
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
    
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool,
        TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2]),
        TabularCPD('Amp', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
        TabularCPD('Temp', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3])
    )
    return VariableElimination(model)

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🔩 Twin Settings")
    mat_name = st.selectbox("Werkstoff wählen", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc - Geschwindigkeit [m/min]", 20, 500, 160)
    f = st.slider("f - Vorschub [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    sim_speed = st.select_slider("Sim-Frequenz", options=[100, 50, 10, 0], value=50)

# --- 5. CALCULATION LOOP ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Physik
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc = (fc * d) / 2000
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 400)
    s['wear'] += wear_inc
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    amp = (0.005 + (s['wear'] * 0.003)) * (1 + s['seed'].normal(0, 0.1))
    
    # KI-Inferenz
    engine = get_engine()
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 200 else (1 if s['cycle'] < 600 else 2),
        'Load': 1 if mc > (d * 2.2) else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if cooling else 1
    }).values[2]
    
    if risk > 0.98: s['broken'] = True; s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc})

# --- 6. INTERAKTIVE PREDICTIVE ANZEIGE ---
st.title("🔩 AI Twin v20.3: Predictive Maintenance Edition")

col_main, col_pred = st.columns([2, 1])

with col_pred:
    st.subheader("🔮 Predictive Analytics")
    if st.session_state.twin['history']:
        # Berechnung der Time-To-Failure (TTF)
        hist = st.session_state.twin['history']
        if len(hist) > 5:
            recent_risk = [h['r'] for h in hist[-5:]]
            trend = np.polyfit(range(5), recent_risk, 1)[0]
            if trend > 0:
                ttf = int((0.8 - hist[-1]['r']) / trend)
                ttf = max(0, ttf)
            else:
                ttf = 999
        else:
            ttf = "Rechne..."
        
        status_text = "SICHER" if risk < 0.4 else ("WARNUNG" if risk < 0.8 else "KRITISCH")
        status_class = "status-ok" if risk < 0.4 else ("status-warn" if risk < 0.8 else "status-crit")
        
        st.markdown(f"""
            <div class="predictive-card">
                <span class="sub-label">Zyklen bis Wartung</span><br>
                <div class="ttf-value">{ttf}</div>
                <span class="{status_class}">STATUS: {status_text}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(f"Das System empfiehlt: {'Weiterarbeiten' if risk < 0.5 else 'Vorschub reduzieren oder Werkzeug wechseln'}")
    else:
        st.write("Warten auf Prozessstart...")

with col_main:
    # Risiko-Graph wie bisher
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#f85149')))
        fig.update_layout(height=250, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

# Buttons
c1, c2, c3 = st.columns(3)
if c1.button("▶️ START / STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if c2.button("🔄 RESET", use_container_width=True): 
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
    st.rerun()

# Automatischer Rerun
if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
