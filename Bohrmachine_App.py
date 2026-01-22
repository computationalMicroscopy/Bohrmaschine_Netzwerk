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
st.set_page_config(layout="wide", page_title="KI - Bohrsystem XAI Twin", page_icon="⚙️")

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
    .xai-label { font-size: 0.75rem; color: #8b949e; margin-bottom: 2px; }
    .xai-bar-bg { background: #30363d; border-radius: 5px; height: 8px; width: 100%; margin-bottom: 10px; }
    .xai-bar-fill { height: 8px; border-radius: 5px; transition: width 0.5s; }
    .predictive-card {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.2) 0%, rgba(5, 7, 10, 0.8) 100%);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 5px 0; }
    .red-glow { color: #f85149; text-shadow: 0 0 15px rgba(248, 81, 73, 0.5); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-ENGINE ---
@st.cache_resource
def get_engine():
    model = BayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State')])
    # CPDs definieren (wie zuvor)
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
                    if score <= 3: v = [0.99, 0.005, 0.005]
                    elif score <= 8: v = [0.60, 0.35, 0.05]
                    elif score <= 12: v = [0.15, 0.45, 0.40]
                    elif score <= 16: v = [0.05, 0.15, 0.80]
                    else: v = [0.01, 0.04, 0.95]
                    z_matrix.append(v)
    cpd_state = TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2])
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state)
    return VariableElimination(model)

def categorize_states(cycle, wear, temp, md, mat_crit, cooling_active, sens_load):
    age_cat = 0 if cycle < 250 else (1 if cycle < 650 else 2)
    load_cat = 1 if md > ((12.0 * 2.2) / sens_load) else 0
    therm_cat = 1 if temp > mat_crit else 0
    cool_cat = 0 if cooling_active else 1
    return age_cat, load_cat, therm_cat, cool_cat

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
                             't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Live-Schnittdaten")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 500, 160)
    f = st.slider("f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlung aktiv", value=True)
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)
    sim_speed = st.select_slider("Speed", options=[200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK ---
engine = get_engine()
s = st.session_state.twin

if s['active'] and not s['broken']:
    s['cycle'] += 1
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300))
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.4)

    a_c, l_c, t_c, c_c = categorize_states(s['cycle'], s['wear'], s['t_current'], mc_raw, mat['temp_crit'], cooling, sens_load)
    s['risk'] = engine.query(['State'], evidence={'Age': a_c, 'Load': l_c, 'Therm': t_c, 'Cool': c_c}).values[2]
    
    if s['risk'] > 0.98: s['broken'] = True; s['active'] = False
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'mc': mc_raw, 'ac': a_c, 'lc': l_c, 'tc': t_c, 'cc': c_c})

# --- 6. UI ---
st.title("KI - BOHRSYSTEM | XAI Digital Twin")
col_metrics, col_main, col_xai = st.columns([1, 2, 1])

with col_metrics:
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{s["wear"]:.1f} %</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{s["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)

with col_main:
    if len(s['history']) > 0:
        df_p = pd.DataFrame(s['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r'] * 100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['mc'], name="Moment [Nm]", line=dict(color='#58a6ff')), row=2, col=1)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_xai:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.subheader("🕵️ XAI Analyse")
    st.caption("Einflussfaktoren auf das aktuelle Risiko:")
    
    # Aktuelle Zustände abrufen
    cur_mc = (mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2) * d) / 2000
    a_c, l_c, t_c, c_c = categorize_states(s['cycle'], s['wear'], s['t_current'], cur_mc, mat['temp_crit'], cooling, sens_load)
    
    factors = [
        ("Werkzeugalter", a_c / 2.0, "#58a6ff"),
        ("Mechanische Last", l_c * 1.0, "#e3b341"),
        ("Thermische Last", t_c * 1.0, "#f85149"),
        ("Kühlung (Fehlend)", c_c * 1.0, "#bc8cff")
    ]
    
    for label, val, color in factors:
        st.markdown(f"""
            <div class="xai-label">{label}</div>
            <div class="xai-bar-bg"><div class="xai-bar-fill" style="width: {val*100}%; background: {color};"></div></div>
        """, unsafe_allow_html=True)
    
    st.divider()
    if s['risk'] < 0.2: st.info("✅ System im stabilen Bereich.")
    elif s['risk'] < 0.7: st.warning("⚠️ Erhöhte Parameter-Abweichung!")
    else: st.error("🚨 KRITISCH: Bruchgefahr!")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
if c2.button("🔄 RESET", use_container_width=True):
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_speed / 1000); st.rerun()
