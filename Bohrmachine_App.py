import streamlit as st
import pandas as pd
import numpy as np
# Robuster Import-Block für verschiedene pgmpy Versionen
try:
    from pgmpy.models import BayesianNetwork
except ImportError:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & DESIGN (Original UI) ---
st.set_page_config(layout="wide", page_title="KI-Zwilling Bohrsystem v21.8.2", page_icon="⚙️")

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

# --- 2. KI-ENGINE ---
@st.cache_resource
def get_engine():
    model = BayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State')])
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.95], [0.05]])

    z_matrix = []
    for age in range(3):
        for load in range(2):
            for therm in range(2):
                for cool in range(2):
                    # Dynamische Inferenz-Logik: Thermik und Kühlung reagieren extrem stark
                    score = (age * 2) + (load * 4) + (therm * 7) + (cool * 8)
                    if score <= 3: v = [0.99, 0.005, 0.005]
                    elif score <= 8: v = [0.60, 0.35, 0.05]
                    elif score <= 12: v = [0.15, 0.45, 0.40]
                    elif score <= 16: v = [0.05, 0.15, 0.80]
                    else: v = [0.01, 0.04, 0.95]
                    z_matrix.append(v)

    cpd_state = TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2])
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state)
    return VariableElimination(model)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
                             't_current': 22.0, 'seed': np.random.RandomState(42)}

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Inconel": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Parameter")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    st.divider()
    cycle_step = st.number_input("Schrittweite", 1, 50, 1)
    sim_speed = st.select_slider("Verzögerung (ms)", options=[500, 200, 100, 50, 10, 0], value=50)
    st.divider()
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)

# --- 5. LOGIK (PHYSIK & KI) ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += cycle_step

    # Physik-Berechnung
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    # Dynamischer Verschleiß basierend auf vc
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300)) * cycle_step
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    # Thermische Trägheit mit Realismus-Rauschen
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.4)

    # KI-Kategorisierung (Trigger)
    age_cat = 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2)
    load_cat = 1 if mc_raw > ((d * 2.2) / sens_load) else 0
    therm_cat = 1 if s['t_current'] > mat['temp_crit'] else 0
    cool_cat = 0 if cooling else 1

    engine = get_engine()
    risk = engine.query(['State'], evidence={'Age': age_cat, 'Load': load_cat, 'Therm': therm_cat, 'Cool': cool_cat}).values[2]

    if risk > 0.98 or s['wear'] > 100: s['broken'] = True; s['active'] = False

    zeit = time.strftime("%H:%M:%S")
    s['history'].append({'c': s['cycle'], 'r': risk, 'w': s['wear'], 't': s['t_current'], 'mc': mc_raw})
    s['logs'].insert(0, f"[{zeit}] ZYK {s['cycle']} | RISIKO: {risk:.1%} | KI-Zustand: [A:{age_cat} L:{load_cat} T:{therm_cat} C:{cool_cat}]")

# --- 6. UI ---
st.title("KI-ZWILLING | BOHRSYSTEM v21.8.2")
col_metrics, col_main, col_logs = st.columns([1, 2, 1])

with col_metrics:
    st.markdown(f'<div class="glass-card"><span class="val-title">Zyklus</span><br><span class="val-main blue-glow">{st.session_state.twin["cycle"]}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{st.session_state.twin["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{st.session_state.twin["wear"]:.1f} %</span></div>', unsafe_allow_html=True)

with col_main:
    ttf = "---"
    if len(st.session_state.twin['history']) > 3:
        df_calc = pd.DataFrame(st.session_state.twin['history'][-15:])
        z = np.polyfit(df_calc['c'], df_calc['w'], 1)
        ttf = max(0, int((100 - st.session_state.twin['wear']) / max(0.000001, z[0])))
    st.markdown(f'<div class="predictive-card"><span class="val-title">🔮 Predictive Maintenance TTF</span><br><div class="ttf-val">{ttf}</div><span class="val-title">Zyklen bis Wartung</span></div>', unsafe_allow_html=True)
    
    if len(st.session_state.twin['history']) > 0:
        df_p = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r']*100, fill='tozeroy', name="Risiko %", line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['mc'], name="Md [Nm]", line=dict(color='#58a6ff')), row=2, col=1)
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_logs:
    st.markdown('<p class="val-title">Echtzeit-Analyse (XAI)</p>', unsafe_allow_html=True)
    log_txt = "".join([f"<div style='margin-bottom:8px; border-bottom:1px solid #30363d; padding-bottom:4px; color:#3fb950; font-family:monospace;'>{l}</div>" for l in st.session_state.twin['logs'][:40]])
    st.markdown(f'<div class="terminal">{log_txt}</div>', unsafe_allow_html=True)

st.divider()
col_btn1, col_btn2 = st.columns(2)
if col_btn1.button("▶ START / STOPP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if col_btn2.button("🔄 RESET", use_container_width=True):
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed / 1000); st.rerun()
