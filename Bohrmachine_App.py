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
st.set_page_config(layout="wide", page_title="AI Precision Twin v20.5 Pro", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-container { 
        background-color: #161b22; border-left: 5px solid #58a6ff; 
        border-radius: 8px; padding: 15px; margin: 5px;
    }
    .predictive-alert { 
        background: linear-gradient(135deg, #21262d 0%, #0d1117 100%);
        border: 2px solid #e3b341; border-radius: 10px; padding: 20px; text-align: center;
    }
    .main-cycle { font-family: 'JetBrains Mono', monospace; font-size: 3rem; color: #79c0ff; font-weight: 800; }
    .ttf-value { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; font-weight: bold; }
    .sub-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .log-terminal { font-family: 'Consolas', monospace; font-size: 0.8rem; height: 300px; overflow-y: auto; background: #010409; padding: 15px; border: 1px solid #30363d; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION MANAGEMENT ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'seed': np.random.RandomState(42)
    }

# --- 3. KI-KERN & MATERIAL-DATEN ---
MATERIALIEN = {
    "Stahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan (TiAl6V4)": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel 718": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([
        ('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'),
        ('State', 'Amp'), ('State', 'Temp')
    ])
    cpds = [
        TabularCPD('Age', 3, [[0.33], [0.33], [0.34]]),
        TabularCPD('Load', 2, [[0.8], [0.2]]),
        TabularCPD('Therm', 2, [[0.9], [0.1]]),
        TabularCPD('Cool', 2, [[0.98], [0.02]])
    ]
    z_matrix = []
    for a in range(3):
        for l in range(2):
            for t in range(2):
                for c in range(2):
                    score = (a * 2) + (l * 4) + (t * 5) + (c * 7)
                    p2 = min(0.99, (score**2.5) / 350.0)
                    p1 = min(1.0-p2, score / 15.0)
                    z_matrix.append([1.0-p1-p2, p1, p2])
    model.add_cpds(*cpds,
        TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2]),
        TabularCPD('Amp', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
        TabularCPD('Temp', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3])
    )
    return VariableElimination(model)

# --- 4. SIDEBAR - BEDIENELEMENTE ---
with st.sidebar:
    st.header("⚙️ Maschinensteuerung")
    mat_name = st.selectbox("Werkstoff wählen", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc - Geschwindigkeit [m/min]", 20, 500, 160)
    f = st.slider("f - Vorschub [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung EIN", value=True)
    sim_speed = st.select_slider("Simulations-Tempo", options=[100, 50, 10, 0], value=50)

# --- 5. PHYSIK-ENGINE & KI ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Kienzle & Taylor Physik
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc = (fc * d) / 2000
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 400)
    s['wear'] += wear_inc
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    amp = (0.005 + (s['wear'] * 0.003)) * (1 + s['seed'].normal(0, 0.1))
    
    # Inferenz
    engine = get_engine()
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2),
        'Load': 1 if mc > (d * 2.2) else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if cooling else 1
    }).values[2]
    
    if risk > 0.98: s['broken'] = True; s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc})
    s['logs'].insert(0, f"CYC {s['cycle']:03d} | Md: {mc:.2f}Nm | Risk: {risk:.1%}")

# --- 6. MAIN DASHBOARD ---
st.title("🔩 Industrial Precision Twin v20.5 Predictive")

col_main, col_side = st.columns([2, 1])

with col_side:
    # --- INTERAKTIVE FRÜHWARNANZEIGE ---
    st.subheader("🔮 Frühwarnsystem")
    if st.session_state.twin['history']:
        hist = st.session_state.twin['history']
        current_risk = hist[-1]['r']
        
        # Prädiktive TTF-Logik
        if len(hist) > 5:
            trend = np.polyfit(range(5), [h['r'] for h in hist[-5:]], 1)[0]
            ttf = int((0.8 - current_risk) / trend) if trend > 0 else 999
            ttf = max(0, ttf)
        else: ttf = "Initialisierung..."
        
        st.markdown(f"""
            <div class="predictive-alert">
                <span class="sub-label">Sichere Restlaufzeit</span><br>
                <div class="ttf-value">{ttf} Zyklen</div>
                <span style="color: {'#3fb950' if current_risk < 0.5 else '#e3b341'}">
                    {"SYSTEM STABIL" if current_risk < 0.5 else "WARTUNG EMPFOHLEN"}
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="sub-label" style="margin-top:20px">System-Logbuch</p>', unsafe_allow_html=True)
    log_txt = "".join([f"<div>{l}</div>" for l in st.session_state.twin['logs'][:50]])
    st.markdown(f'<div class="log-terminal">{log_txt}</div>', unsafe_allow_html=True)

with col_main:
    # Metriken
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-container"><span class="sub-label">Aktueller Zyklus</span><br><div class="main-cycle">{st.session_state.twin["cycle"]}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-container"><span class="sub-label">Temperatur (°C)</span><br><div class="main-cycle" style="color:#f85149">{st.session_state.twin["t_current"]:.1f}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-container"><span class="sub-label">Last (Drehmoment)</span><br><div class="main-cycle" style="color:#58a6ff">{st.session_state.twin["history"][-1]["mc"] if st.session_state.twin["history"] else 0.0:.2f}</div></div>', unsafe_allow_html=True)
    
    # Risiko-Visualisierung
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#f85149', width=3)))
        fig.update_layout(height=380, template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# Steuerung
if st.button("▶️ PROZESS START / STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if st.button("🔄 VOLLSTÄNDIGER RESET", use_container_width=True):
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
