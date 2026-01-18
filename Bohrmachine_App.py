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
st.set_page_config(layout="wide", page_title="AI Twin v20.8 Education", page_icon="📖")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { 
        background-color: #161b22; border-left: 4px solid #58a6ff; 
        border-radius: 6px; padding: 12px; margin-bottom: 10px;
    }
    .main-val { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; color: #79c0ff; font-weight: 700; }
    .sub-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .glossary-box { background-color: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-top: 20px; }
    .log-terminal { font-family: 'Consolas', monospace; font-size: 0.75rem; height: 350px; overflow-y: auto; background: #010409; padding: 10px; border: 1px solid #30363d; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION MANAGEMENT ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'seed': np.random.RandomState(42)
    }

# --- 3. KI-KERN & PHYSIK ---
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

# --- 4. SIDEBAR - BEDIENUNG MIT ERKLÄRUNGEN ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()), help="Bestimmt die spezifische Schnittkraft (kc1.1) und die thermische Belastbarkeit.")
    mat = MATERIALIEN[mat_name]
    
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160, help="Die Geschwindigkeit, mit der die Schneide am Werkstück entlangfährt. Haupteinflussfaktor für Hitze.")
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18, help="Weg des Bohrers pro Umdrehung. Bestimmt die Spandicke und die mechanische Last.")
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0, help="Durchmesser des Werkzeugs. Beeinflusst das Drehmoment maßgeblich.")
    
    cooling = st.toggle("Kühlschmierung", value=True, help="Aktiviert die Wärmeabfuhr. Senkt das Risiko für thermisches Versagen.")
    
    st.divider()
    st.header("📡 Sensor-Empfindlichkeit")
    sens_vib = st.slider("Vibrations-Sensitivität", 0.1, 5.0, 1.0, help="Verstärkungsfaktor für das Vibrationssignal. Höher = Frühere Warnung bei Rattern.")
    sens_load = st.slider("Last-Sensitivität", 0.1, 5.0, 1.0, help="Gewichtung des Drehmoments in der KI-Logik.")
    
    sim_speed = st.select_slider("Abtastung (ms)", options=[100, 50, 10, 0], value=50)

# --- 5. BERECHNUNG ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc_raw = (fc * d) / 2000
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 400)
    s['wear'] += wear_inc
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    
    noise = s['seed'].normal(0, 0.001) * sens_vib
    amp = ((0.005 + (s['wear'] * 0.002)) * sens_vib) + noise
    
    engine = get_engine()
    load_threshold = (d * 2.2) / sens_load
    
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2),
        'Load': 1 if mc_raw > load_threshold else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if cooling else 1
    }).values[2]
    
    if risk > 0.98: s['broken'] = True; s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc_raw})
    s['logs'].insert(0, f"CYC {s['cycle']:03d} | Risk: {risk:.1%} | Amp: {amp:.4f} | Md: {mc_raw:.1f}Nm")

# --- 6. DASHBOARD ---
st.title("Industrial Precision Twin v20.8 📖")

col_metrics, col_graph, col_logs = st.columns([0.6, 1.8, 1.0])

with col_metrics:
    st.markdown(f'<div class="metric-card" title="Anzahl der durchgeführten Bohrungen/Operationen."><span class="sub-label">Zyklus</span><br><div class="main-val">{st.session_state.twin["cycle"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card" title="Aktuelle Temperatur an der Schneide. Kritisch ab materialabhängigem Temp-Limit."><span class="sub-label">Temperatur</span><br><div class="main-val" style="color:#f85149">{st.session_state.twin["t_current"]:.1f}°C</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card" title="Drehmoment (Last) in Newtonmeter. Berechnet nach der Kienzle-Formel."><span class="sub-label">Drehmoment</span><br><div class="main-val" style="color:#58a6ff">{st.session_state.twin["history"][-1]["mc"] if st.session_state.twin["history"] else 0.0:.1f}Nm</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card" title="RMS-Vibrationswert. Steigt bei Werkzeugstumpfung oder Instabilität."><span class="sub-label">Vibration (RMS)</span><br><div class="main-val" style="color:#3fb950">{st.session_state.twin["history"][-1]["amp"] if st.session_state.twin["history"] else 0.0:.4f}</div></div>', unsafe_allow_html=True)

    # KLEINES GLOSSAR UNTER DEN METRIKEN
    with st.container():
        st.markdown("""
        <div class="glossary-box">
        <p style="color:#58a6ff; font-weight:bold; margin-bottom:5px;">Kurz-Glossar:</p>
        <small><b>vc:</b> Schnittgeschwindigkeit. Treibt die Temperatur.<br>
        <b>f:</b> Vorschub. Treibt die mechanische Kraft.<br>
        <b>kc1.1:</b> Spez. Schnittkraft des Materials.<br>
        <b>Bayes-Inferenz:</b> Wahrscheinlichkeits-Berechnung des Zustands.</small>
        </div>
        """, unsafe_allow_html=True)

with col_graph:
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("KI-Bruchrisiko (%)", "Sensor-Rohdaten (Last & Vibration)"))
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="Bruchrisiko", line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Last (Nm)", line=dict(color='#58a6ff')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['amp']*1000, name="Vibration", line=dict(color='#3fb950')), row=2, col=1)
        fig.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

with col_logs:
    st.markdown('<p class="sub-label">KI- & Sensor-Protokoll</p>', unsafe_allow_html=True)
    log_content = "".join([f"<div>{l}</div>" for l in st.session_state.twin['logs'][:100]])
    st.markdown(f'<div class="log-terminal">{log_content}</div>', unsafe_allow_html=True)

st.divider()
c1, c2 = st.columns(2)
if c1.button("▶️ PROZESS START / STOP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if c2.button("🔄 SYSTEM-RESET", use_container_width=True):
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
