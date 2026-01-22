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
st.set_page_config(layout="wide", page_title="KI - Bohrsystem Digital Twin v3.0", page_icon="⚙️")

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
    .sandbox-card {
        background: rgba(63, 185, 80, 0.05);
        border: 1px dashed #3fb950; border-radius: 15px; padding: 20px;
    }
    .xai-bar-bg { background: #30363d; border-radius: 5px; height: 8px; width: 100%; margin-bottom: 10px; }
    .xai-bar-fill { height: 8px; border-radius: 5px; transition: width 0.5s; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 5px 0; }
    .blue-glow { color: #58a6ff; text-shadow: 0 0 15px rgba(88, 166, 255, 0.5); }
    .red-glow { color: #f85149; text-shadow: 0 0 15px rgba(248, 81, 73, 0.5); }
    .green-glow { color: #3fb950; text-shadow: 0 0 15px rgba(63, 185, 80, 0.5); }
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

def categorize(cycle, wear, temp, md, mat_crit, cooling_active, sens_load):
    a_c = 0 if cycle < 250 else (1 if cycle < 650 else 2)
    l_c = 1 if md > ((12.0 * 2.2) / sens_load) else 0
    t_c = 1 if temp > mat_crit else 0
    c_c = 0 if cooling_active else 1
    return a_c, l_c, t_c, c_c

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
                             't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Live-Prozess")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 500, 160)
    f = st.slider("f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlung aktiv", value=True)
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)
    sim_speed = st.select_slider("Sim-Speed", options=[200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK ---
engine = get_engine()
s = st.session_state.twin

# Aktuelle physikalische Werte (immer berechnet)
cur_fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
cur_md = (cur_fc * d) / 2000

if s['active'] and not s['broken']:
    s['cycle'] += 1
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300))
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.4)
    
    a, l, t, c = categorize(s['cycle'], s['wear'], s['t_current'], cur_md, mat['temp_crit'], cooling, sens_load)
    s['risk'] = engine.query(['State'], evidence={'Age': a, 'Load': l, 'Therm': t, 'Cool': c}).values[2]
    
    if s['risk'] > 0.98: s['broken'] = True; s['active'] = False
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'md': cur_md})

# --- 6. UI ---
st.title("KI - BOHRSYSTEM | Digitaler Zwilling v3.0")
col_metrics, col_main, col_whatif = st.columns([1, 2, 1])

with col_metrics:
    st.markdown(f'<div class="glass-card"><span class="val-title">Real-Time Risiko</span><br><span class="val-main {"red-glow" if s["risk"]>0.5 else "blue-glow"}">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{s["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Moment</span><br><span class="val-main blue-glow">{cur_md:.1f} Nm</span></div>', unsafe_allow_html=True)

with col_main:
    # Graphik
    if len(s['history']) > 0:
        df_p = pd.DataFrame(s['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r'] * 100, fill='tozeroy', name="Ist-Risiko %", line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['w'], name="Verschleiß %", line=dict(color='#e3b341')), row=2, col=1)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_whatif:
    st.markdown('<div class="sandbox-card">', unsafe_allow_html=True)
    st.subheader("🧪 Strategie-Planer")
    st.caption("Ändere Parameter hypothetisch:")
    
    h_vc = st.slider("Hypoth. vc", 20, 500, vc)
    h_f = st.slider("Hypoth. f", 0.02, 1.0, f)
    h_cool = st.toggle("Hypoth. Kühlung", value=cooling)
    
    # Hypothesen-Inferenz
    h_fc = mat['kc1.1'] * (h_f ** (1 - mat['mc'])) * (d / 2)
    h_md = (h_fc * d) / 2000
    h_temp = 22 + (s['wear'] * 1.5) + (h_vc * 0.2) + (0 if h_cool else 250)
    
    ha, hl, ht, hc = categorize(s['cycle'], s['wear'], h_temp, h_md, mat['temp_crit'], h_cool, sens_load)
    h_risk = engine.query(['State'], evidence={'Age': ha, 'Load': hl, 'Therm': ht, 'Cool': hc}).values[2]
    
    # Delta-Analyse
    diff = h_risk - s['risk']
    st.metric("Plan-Risiko", f"{h_risk:.1%}", delta=f"{diff:.1%}", delta_color="inverse")
    
    # KI-Empfehlung
    st.divider()
    st.markdown("**KI-Vorschlag:**")
    if h_risk > 0.4:
        st.error("🚨 Strategie unsicher! Senken Sie vc oder f.")
    elif diff < -0.05:
        st.success("✅ Optimale Rettungsstrategie!")
    else:
        st.info("ℹ️ Plan entspricht aktuellem Risiko.")
        
    # Explainable AI Bars für den PLAN
    st.caption("Einflussfaktoren im Plan:")
    for lbl, v, clr in [("Alter", ha/2, "#58a6ff"), ("Last", hl, "#e3b341"), ("Hitze", ht, "#f85149")]:
        st.markdown(f'<div style="font-size:10px">{lbl}</div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{v*100}%; background:{clr}"></div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
if c2.button("🔄 RESET", use_container_width=True):
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_speed / 1000); st.rerun()
