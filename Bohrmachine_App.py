import streamlit as st
import pandas as pd
import numpy as np
import time

# Import-Fix für pgmpy 0.1.25+ 
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. SETUP & DESIGN ---
st.set_page_config(layout="wide", page_title="KI - Bohrmaschinen Digital Twin v2.5", page_icon="⚙️")

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
    .sandbox-card {
        background: rgba(63, 185, 80, 0.1);
        border: 1px dashed #3fb950; border-radius: 15px; padding: 15px;
    }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 5px 0; }
    .blue-glow { color: #58a6ff; text-shadow: 0 0 15px rgba(88, 166, 255, 0.5); }
    .red-glow { color: #f85149; text-shadow: 0 0 15px rgba(248, 81, 73, 0.5); }
    .green-glow { color: #3fb950; text-shadow: 0 0 15px rgba(63, 185, 80, 0.5); }
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
                             't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Echtzeit-Parameter")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    st.divider()
    sim_speed = st.select_slider("Sim-Speed (Verzögerung ms)", options=[500, 200, 100, 50, 10, 0], value=50)
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)

# --- 5. LOGIK ---
engine = get_engine()

if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1

    # Physik
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300))
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.4)

    # Inferenz
    age_cat = 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2)
    load_cat = 1 if mc_raw > ((d * 2.2) / sens_load) else 0
    therm_cat = 1 if s['t_current'] > mat['temp_crit'] else 0
    cool_cat = 0 if cooling else 1

    s['risk'] = engine.query(['State'], evidence={'Age': age_cat, 'Load': load_cat, 'Therm': therm_cat, 'Cool': cool_cat}).values[2]

    if s['risk'] > 0.98 or s['wear'] > 100: 
        s['broken'] = True
        s['active'] = False

    zeit = time.strftime("%H:%M:%S")
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'mc': mc_raw})
    s['logs'].insert(0, f"[{zeit}] ZYK {s['cycle']} | RISIKO: {s['risk']:.1%} | {mat_name}")

# --- 6. UI ---
st.title("KI - BOHRSYSTEM | Digitaler Zwilling v2.5")
col_metrics, col_main, col_logs = st.columns([1, 2, 1])

with col_metrics:
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{st.session_state.twin["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{st.session_state.twin["wear"]:.1f} %</span></div>', unsafe_allow_html=True)
    
    # NEU: WHAT-IF SANDBOX
    st.markdown('<div class="sandbox-card">', unsafe_allow_html=True)
    st.subheader("🧪 What-If Analyse")
    st.caption("Wie würde sich das Risiko ändern, wenn wir...")
    hypo_vc = st.slider("Hypoth. vc", 20, 500, vc, key="h_vc")
    hypo_cool = st.toggle("Hypoth. Kühlung", value=cooling, key="h_cool")
    
    # Hypothesen-Berechnung
    h_fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    h_mc = (h_fc * d) / 2000
    h_t = 22 + (st.session_state.twin['wear'] * 1.5) + (hypo_vc * 0.2) + (0 if hypo_cool else 250)
    
    h_load_cat = 1 if h_mc > ((d * 2.2) / sens_load) else 0
    h_therm_cat = 1 if h_t > mat['temp_crit'] else 0
    h_cool_cat = 0 if hypo_cool else 1
    
    hypo_risk = engine.query(['State'], evidence={'Age': age_cat, 'Load': h_load_cat, 'Therm': h_therm_cat, 'Cool': h_cool_cat}).values[2]
    diff = hypo_risk - st.session_state.twin['risk']
    diff_color = "red" if diff > 0 else "#3fb950"
    
    st.metric("Hypothetisches Risiko", f"{hypo_risk:.1%}", delta=f"{diff:.1%}", delta_color="inverse" if diff < 0 else "normal")
    if diff < -0.1:
        st.success("💡 Maßnahme empfohlen!")
    st.markdown('</div>', unsafe_allow_html=True)

with col_main:
    ttf = "---"
    if len(st.session_state.twin['history']) > 3:
        df_calc = pd.DataFrame(st.session_state.twin['history'][-15:])
        z = np.polyfit(df_calc['c'], df_calc['w'], 1)
        ttf = max(0, int((100 - st.session_state.twin['wear']) / max(0.000001, z[0])))
    st.markdown(f'<div class="predictive-card"><span class="val-title">🔮 TTF (Zyklen bis Ausfall)</span><br><div class="ttf-val">{ttf}</div></div>', unsafe_allow_html=True)

    if len(st.session_state.twin['history']) > 0:
        df_p = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r'] * 100, fill='tozeroy', name="Echtzeit-Risiko %", line=dict(color='#f85149')), row=1, col=1)
        # NEU: Hypothesen-Linie im Chart
        fig.add_hline(y=hypo_risk*100, line_dash="dash", line_color="#3fb950", annotation_text="What-If Level", row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['mc'], name="Md [Nm]", line=dict(color='#58a6ff')), row=2, col=1)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_logs:
    st.markdown('<p class="val-title">KI-Analyse Log</p>', unsafe_allow_html=True)
    log_txt = "".join([f"<div style='margin-bottom:8px; border-bottom:1px solid #30363d; padding-bottom:4px; color:#3fb950; font-family:monospace;'>{l}</div>" for l in st.session_state.twin['logs'][:30]])
    st.markdown(f'<div class="terminal">{log_txt}</div>', unsafe_allow_html=True)

st.divider()
c1, c2, c3 = st.columns(3)
if c1.button("▶ START / STOPP", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if c2.button("🔄 RESET", use_container_width=True):
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0}
    st.rerun()
if c3.button("📥 DATEN EXPORT", use_container_width=True):
    if st.session_state.twin['history']:
        pd.DataFrame(st.session_state.twin['history']).to_csv("bohrmaschine_daten.csv")
        st.success("Exportiert als bohrmaschine_daten.csv")

if st.session_state.twin['active']:
    time.sleep(sim_speed / 1000); st.rerun()
