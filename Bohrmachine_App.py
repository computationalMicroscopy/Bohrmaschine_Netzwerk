import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & ULTRA-MODERN THEME ---
st.set_page_config(layout="wide", page_title="AI Precision Twin v21.0 PREDICTIVE", page_icon="🔮")

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

# --- 2. LOGIK-KERN ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42)}

MATERIALIEN = {
    "Stahl 42CrMo4": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl 1.4404": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel Superalloy": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'), ('State', 'Amp'), ('State', 'Temp')])
    cpds = [TabularCPD('Age', 3, [[0.33], [0.33], [0.34]]), TabularCPD('Load', 2, [[0.8], [0.2]]), TabularCPD('Therm', 2, [[0.9], [0.1]]), TabularCPD('Cool', 2, [[0.98], [0.02]])]
    z_matrix = []
    for a in range(3):
        for l in range(2):
            for t in range(2):
                for c in range(2):
                    score = (a * 2) + (l * 4) + (t * 5) + (c * 7)
                    p2 = min(0.99, (score**2.5) / 350.0); p1 = min(1.0-p2, score / 15.0)
                    z_matrix.append([1.0-p1-p2, p1, p2])
    model.add_cpds(*cpds, TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2]),
                   TabularCPD('Amp', 2, [[0.95, 0.4, 0.05], [0.05, 0.6, 0.95]], ['State'], [3]),
                   TabularCPD('Temp', 2, [[0.98, 0.3, 0.02], [0.02, 0.7, 0.98]], ['State'], [3]))
    return VariableElimination(model)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Vision Config")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()), help="Beeinflusst kc1.1 und die Verschleißrate.")
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 500, 160, help="Schnittgeschwindigkeit (Hitze)")
    f = st.slider("f [mm/U]", 0.02, 1.0, 0.18, help="Vorschub (Mechanische Last)")
    d = st.number_input("Ø Werkzeug [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlsystem aktiv", value=True)
    st.divider()
    sens_vib = st.slider("Vibrations-Gain", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.1, 5.0, 1.0)

# --- 4. ENGINE RUN ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc_raw = (fc * d) / 2000
    wear_inc = (mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 400)
    s['wear'] += wear_inc
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    amp = ((0.005 + (s['wear'] * 0.002)) * sens_vib) + s['seed'].normal(0, 0.001) * sens_vib
    engine = get_engine()
    risk = engine.query(['State'], evidence={'Age': 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2),
                                             'Load': 1 if mc_raw > ((d * 2.2) / sens_load) else 0,
                                             'Therm': 1 if s['t_current'] > mat['temp_crit'] else 0,
                                             'Cool': 0 if cooling else 1}).values[2]
    if risk > 0.98 or s['wear'] > 100: s['broken'] = True; s['active'] = False
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc_raw})
    s['logs'].insert(0, f"CYC {s['cycle']} | Status: {'OK' if risk < 0.5 else 'WARN'} | Wear: {s['wear']:.1f}%")

# --- 5. VISUAL INTERFACE ---
st.title("AI PRECISION TWIN | PREDICTIVE MONITORING")

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown(f'<div class="glass-card"><span class="val-title">Zyklus</span><br><span class="val-main blue-glow">{st.session_state.twin["cycle"]}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main red-glow">{st.session_state.twin["t_current"]:.1f}°</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{st.session_state.twin["wear"]:.1f}%</span></div>', unsafe_allow_html=True)

with col_mid:
    # --- PREDICTIVE MAINTENANCE CARD ---
    if st.session_state.twin['history'] and len(st.session_state.twin['history']) > 5:
        df = pd.DataFrame(st.session_state.twin['history'])
        # Stabile Trendberechnung (linearer Fit über die letzten 10 Punkte)
        recent = df.tail(10)
        z = np.polyfit(recent['c'], recent['w'], 1)
        slope = max(0.0001, z[0]) # Verschleiß pro Zyklus
        remaining_cycles = int((100 - st.session_state.twin['wear']) / slope)
        ttf = max(0, remaining_cycles)
    else:
        ttf = "---"

    st.markdown(f"""
    <div class="predictive-card">
        <span class="val-title" style="color:#58a6ff">🔮 Voraussichtliche Restlaufzeit (TTF)</span><br>
        <div class="ttf-val">{ttf}</div>
        <span class="val-title">Zyklen bis Wartung erforderlich</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.twin['history']:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#f85149')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Last Nm", line=dict(color='#58a6ff')), row=2, col=1)
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown('<p class="val-title">Live Analytics Stream</p>', unsafe_allow_html=True)
    log_txt = "".join([f"<div style='margin-bottom:4px;'>{l}</div>" for l in st.session_state.twin['logs'][:60]])
    st.markdown(f'<div class="terminal">{log_txt}</div>', unsafe_allow_html=True)

# Footer
st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ START / STOP SYSTEM", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
if c2.button("🔄 VOLLSTÄNDIGER RESET", use_container_width=True):
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(0.05)
    st.rerun()
