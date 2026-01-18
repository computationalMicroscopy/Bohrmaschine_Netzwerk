import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & DESIGN (MODERN DARK THEME) ---
st.set_page_config(layout="wide", page_title="KI-Zwilling Bohrsystem v21.6", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px; padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px); margin-bottom: 15px;
    }
    .predictive-card {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.2) 0%, rgba(5, 7, 10, 0.8) 100%);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .val-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 5px 0; }
    .ttf-val { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; text-shadow: 0 0 20px rgba(227, 179, 65, 0.4); }
    .blue-glow { color: #58a6ff; text-shadow: 0 0 15px rgba(88, 166, 255, 0.5); }
    .red-glow { color: #f85149; text-shadow: 0 0 15px rgba(248, 81, 73, 0.5); }
    .terminal { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; height: 350px; background: #010409; padding: 15px; border-radius: 10px; border: 1px solid #30363d; color: #3fb950; overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALISIERUNG DES SYSTEMS ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 
        'active': False, 'broken': False, 't_current': 22.0, 
        'seed': np.random.RandomState(42)
    }

# MATERIALIEN MIT SPRECHENDEN DEUTSCHEN NAMEN
MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (rostfrei 1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung (hochfest)": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel (Superlegierung)": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'), ('State', 'Amp'), ('State', 'Temp')])
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
                    p2 = min(0.99, (score**2.5) / 350.0); p1 = min(1.0-p2, score / 15.0)
                    z_matrix.append([1.0-p1-p2, p1, p2])
    model.add_cpds(*cpds, 
        TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2]),
        TabularCPD('Amp', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
        TabularCPD('Temp', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3])
    )
    return VariableElimination(model)

# --- 3. SEITENLEISTE (KONFIGURATION) ---
with st.sidebar:
    st.header("⚙️ Maschinen-Parameter")
    mat_name = st.selectbox("Werkstoff wählen", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    
    st.divider()
    st.subheader("⏱️ Simulationstakt")
    sim_speed = st.select_slider("Verzögerung pro Zyklus (ms)", options=[500, 200, 100, 50, 10, 0], value=50)
    
    st.divider()
    st.subheader("📡 Sensor-Gain")
    sens_vib = st.slider("Vibrations-Empfindlichkeit", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)

# --- 4. KERNLOGIK (SIMULATION & KI) ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # PHYSIK-MODELLE (Kienzle-Formel für Drehmoment Md)
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc_raw = (fc * d) / 2000 # Moment in Nm
    
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 400)
    s['wear'] += wear_inc
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    
    # Vibration: Schwinggeschwindigkeit v_rms in mm/s
    amp = ((0.005 + (s['wear'] * 0.002)) * sens_vib * 10) + s['seed'].normal(0, 0.01) * sens_vib
    
    # KI-INFERENZ (Bayesian Network)
    engine = get_engine()
    risk = engine.query(['State'], evidence={
        'Age': 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2),
        'Load': 1 if mc_raw > ((d * 2.2) / sens_load) else 0,
        'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
        'Cool': 0 if cooling else 1
    }).values[2]
    
    if risk > 0.98 or s['wear'] > 100: s['broken'] = True; s['active'] = False
    
    # Datenspeicherung
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc_raw})
    
    # Protokoll-Eintrag
    zeit = time.strftime("%H:%M:%S")
    s['logs'].insert(0, f"[{zeit}] ZYK {s['cycle']} | Risiko: {risk:.1%} | Drehmoment (Md): {mc_raw:.1f} Nm | Vibration: {amp:.2f} mm/s")

# --- 5. DASHBOARD OBERFLÄCHE ---
st.title("KI-ZWILLING | PRÄZISIONS-ÜBERWACHUNG")

col_metrics, col_main, col_logs = st.columns([1, 2, 1])

with col_metrics:
    st.markdown(f'<div class="glass-card"><span class="val-title">Aktueller Zyklus</span><br><span class="val-main blue-glow">{st.session_state.twin["cycle"]}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{st.session_state.twin["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{st.session_state.twin["wear"]:.1f} %</span></div>', unsafe_allow_html=True)

with col_main:
    # PREDICTIVE MAINTENANCE (TTF Berechnung)
    ttf = "---"
    if len(st.session_state.twin['history']) > 3:
        df_calc = pd.DataFrame(st.session_state.twin['history'])
        z = np.polyfit(df_calc['c'], df_calc['w'], 1)
        steigung = max(0.000001, z[0])
        ttf = max(0, int((100 - st.session_state.twin['wear']) / steigung))

    st.markdown(f'<div class="predictive-card"><span class="val-title" style="color:#58a6ff">🔮 Voraussichtliche Restlaufzeit (TTF)</span><br><div class="ttf-val">{ttf}</div><span class="val-title">Zyklen bis Wartung erforderlich</span></div>', unsafe_allow_html=True)

    # Diagramme (nur wenn Daten vorhanden sind)
    if len(st.session_state.twin['history']) > 0:
        df_p = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r']*100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['mc'], name="Drehmoment Md [Nm]", line=dict(color='#58a6ff')), row=2, col=1)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("System im Leerlauf. Klicken Sie auf Start für Live-Analysen.")

with col_logs:
    st.markdown('<p class="val-title">Live-Analyse-Protokoll</p>', unsafe_allow_html=True)
    log_txt = "".join([f"<div style='margin-bottom:6px; border-bottom:1px solid #30363d; padding-bottom:2px; color:#3fb950; font-family:monospace;'>{l}</div>" for l in st.session_state.twin['logs'][:60]])
    st.markdown(f'<div class="terminal">{log_txt}</div>', unsafe_allow_html=True)

# --- 6. STEUERUNG ---
st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ SIMULATION START / STOPP", use_container_width=True): 
    st.session_state.twin['active'] = not st.session_state.twin['active']
if c2.button("🔄 VOLLSTÄNDIGER RESET", use_container_width=True):
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    if sim_speed > 0:
        time.sleep(sim_speed / 1000)
    st.rerun()
