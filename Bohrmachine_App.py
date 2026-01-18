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
st.set_page_config(layout="wide", page_title="AI Twin v20.4 Master Edition", page_icon="🎓")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-container { 
        background-color: #161b22; border-left: 5px solid #58a6ff; 
        border-radius: 8px; padding: 15px; margin: 5px;
    }
    .predictive-card { 
        background: linear-gradient(135deg, #1c2128 0%, #0d1117 100%);
        border: 1px solid #30363d; border-left: 5px solid #e3b341;
        border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 15px;
    }
    .main-cycle { font-family: 'JetBrains Mono', monospace; font-size: 3rem; color: #79c0ff; font-weight: 800; }
    .ttf-value { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; font-weight: bold; }
    .sub-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .log-terminal { font-family: 'Consolas', monospace; font-size: 0.8rem; height: 250px; overflow-y: auto; background: #010409; padding: 15px; border: 1px solid #30363d; border-radius: 5px; }
    .status-ok { color: #3fb950; font-weight: bold; }
    .status-warn { color: #d29922; font-weight: bold; }
    .status-crit { color: #f85149; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION MANAGEMENT ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'sabotage_coolant': False, 'material_anomaly': 1.0, 'seed': np.random.RandomState(42)
    }

# --- 3. TEACHER PANEL ---
with st.expander("🛠️ TEACHER PANEL (Sabotage-Simulation)", expanded=False):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        sab_cool = st.toggle("Kühlsystem manipulieren (Leckage)", value=st.session_state.twin['sabotage_coolant'])
        st.session_state.twin['sabotage_coolant'] = sab_cool
    with col_t2:
        anom = st.slider("Material-Anomalie (Härteeinschluss)", 1.0, 3.0, st.session_state.twin['material_anomaly'])
        st.session_state.twin['material_anomaly'] = anom

# --- 4. MATERIAL-DATENBANK & KI-ENGINE ---
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

# --- 5. SIDEBAR & PHYSIK-INPUT ---
with st.sidebar:
    st.title("🔩 Maschinen-Parameter")
    mat_name = st.selectbox("Werkstoff wählen", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc - Geschwindigkeit [m/min]", 20, 500, 160)
    f = st.slider("f - Vorschub [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling_ui = st.toggle("Kühlschmierung aktiv", value=True)
    sim_speed = st.select_slider("Sim-Frequenz", options=[100, 50, 10, 0], value=50)

# --- 6. CORE CALCULATION ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Physikalische Simulation
    eff_cooling = cooling_ui and not s['sabotage_coolant']
    fc = mat['kc1.1'] * s['material_anomaly'] * (f** (1-mat['mc'])) * (d/2)
    mc = (fc * d) / 2000
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if eff_cooling else 400)
    s['wear'] += wear_inc
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if eff_cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    amp = (0.005 + (s['wear'] * 0.003)) * (1 + s['seed'].normal(0, 0.1))
    
    # KI-Inferenz
    engine = get_engine()
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2),
        'Load': 1 if mc > (d * 2.2) else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if eff_cooling else 1
    }).values[2]
    
    if risk > 0.98: s['broken'] = True; s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc})
    if s['sabotage_coolant']: s['logs'].insert(0, f"⚠️ CYC {s['cycle']} | THERMAL LEAK")
    s['logs'].insert(0, f"CYC {s['cycle']} | Md: {mc:.2f}Nm | Risk: {risk:.1%}")

# --- 7. DASHBOARD LAYOUT ---
st.title("🔩 AI Precision Twin v20.4 Master Edition")

col_left, col_right = st.columns([2, 1])

with col_right:
    # --- PREDICTIVE INTERACTIVE CARD ---
    st.markdown('<p class="sub-label">Predictive Analytics</p>', unsafe_allow_html=True)
    if st.session_state.twin['history']:
        hist = st.session_state.twin['history']
        risk_val = hist[-1]['r']
        # TTF Trend-Berechnung
        if len(hist) > 10:
            y = [h['r'] for h in hist[-10:]]
            trend = np.polyfit(range(10), y, 1)[0]
            ttf = int((0.8 - risk_val) / trend) if trend > 0.0001 else 999
            ttf = max(0, ttf)
        else: ttf = "..."
        
        status_class = "status-ok" if risk_val < 0.4 else ("status-warn" if risk_val < 0.8 else "status-crit")
        st.markdown(f"""
            <div class="predictive-card">
                <span class="sub-label">Zyklen bis Werkzeugwechsel</span><br>
                <div class="ttf-value">{ttf}</div>
                <span class="{status_class}">SYSTEM STATUS: {"STABIL" if risk_val < 0.4 else "WARNUNG"}</span>
            </div>
        """, unsafe_allow_html=True)
    
    # --- LOG TERMINAL ---
    st.markdown('<p class="sub-label">System Logs</p>', unsafe_allow_html=True)
    log_content = "".join([f"<div>{l}</div>" for l in st.session_state.twin['logs'][:50]])
    st.markdown(f'<div class="log-terminal">{log_content}</div>', unsafe_allow_html=True)

with col_left:
    # --- METRICS & MAIN GRAPH ---
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(f'<div class="metric-container"><span class="sub-label">Zyklus</span><br><div class="main-cycle">{st.session_state.twin["cycle"]}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-container"><span class="sub-label">Temperatur</span><br><div class="main-cycle" style="color:#f85149">{st.session_state.twin["t_current"]:.1f}°</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-container"><span class="sub-label">Verschleiß</span><br><div class="main-cycle" style="color:#e3b341">{st.session_state.twin["wear"]:.1f}%</div></div>', unsafe_allow_html=True)
    
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#f85149')))
        fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Last (Nm)", line=dict(color='#58a6ff')), secondary_y=True)
        fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# Controls
if st.button("▶️ PROZESS START / STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if st.button("🔄 SYSTEM-RESET", use_container_width=True):
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'sabotage_coolant':False,'material_anomaly':1.0,'seed':np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
