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
st.set_page_config(layout="wide", page_title="AI Precision Twin v20.9 VISION", page_icon="💎")

st.markdown("""
    <style>
    /* Globales Dark Theme */
    .stApp { background-color: #05070a; color: #e1e4e8; }
    
    /* Glasmorphismus Karten */
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px);
        margin-bottom: 15px;
    }
    
    /* Metrik Styling */
    .val-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.8rem; font-weight: 800; margin: 10px 0; }
    .unit { font-size: 1rem; color: #58a6ff; font-weight: 400; }
    
    /* Status Farben */
    .blue-glow { color: #58a6ff; text-shadow: 0 0 15px rgba(88, 166, 255, 0.5); }
    .red-glow { color: #f85149; text-shadow: 0 0 15px rgba(248, 81, 73, 0.5); }
    .orange-glow { color: #d29922; text-shadow: 0 0 15px rgba(210, 153, 34, 0.5); }
    .green-glow { color: #3fb950; text-shadow: 0 0 15px rgba(63, 185, 80, 0.5); }

    /* Terminal Styling */
    .terminal {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        height: 400px;
        background: #010409;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363d;
        color: #3fb950;
        overflow-y: auto;
    }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(135deg, #1f6feb 0%, #11438f 100%);
        color: white; border: none; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(31, 111, 235, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIK-KERN (Unverändert stabil) ---
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

# --- 3. SIDEBAR (Kontrolle & Glossar) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/cpu.png", width=80)
    st.title("Vision Config")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 500, 160)
    f = st.slider("f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Ø Werkzeug [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlsystem aktiv", value=True)
    
    st.markdown("---")
    st.subheader("📡 Sensor Calibration")
    sens_vib = st.slider("Vibrations-Gain", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.1, 5.0, 1.0)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.1);">
    <b>📖 Info-Quicklinks:</b><br>
    <b>vc:</b> Schnittgeschwindigkeit (Hitze-Treiber)<br>
    <b>f:</b> Vorschub (Kraft-Treiber)<br>
    <b>kc1.1:</b> Spez. Materialwiderstand
    </div>
    """, unsafe_allow_html=True)

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
    if risk > 0.98: s['broken'] = True; s['active'] = False
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'amp':amp, 'mc':mc_raw})
    s['logs'].insert(0, f"[{time.strftime('%H:%M:%S')}] CYC {s['cycle']} - Risk: {risk:.1%} - Load: {mc_raw:.1f}Nm")

# --- 5. VISUAL INTERFACE ---
st.title("AI PRECISION TWIN | VISION ELITE")

col_info, col_chart, col_log = st.columns([0.8, 2, 1])

with col_info:
    # Metriken in Glasmorphismus-Karten
    st.markdown(f"""
    <div class="glass-card">
        <span class="val-title">Prozesszyklus</span><br>
        <span class="val-main blue-glow">{st.session_state.twin['cycle']}</span>
    </div>
    <div class="glass-card">
        <span class="val-title">Thermische Last</span><br>
        <span class="val-main red-glow">{st.session_state.twin['t_current']:.1f}<span class="unit">°C</span></span>
    </div>
    <div class="glass-card">
        <span class="val-title">Mechanische Last</span><br>
        <span class="val-main orange-glow">{st.session_state.twin['history'][-1]['mc'] if st.session_state.twin['history'] else 0.0:.1f}<span class="unit">Nm</span></span>
    </div>
    <div class="glass-card">
        <span class="val-title">Verschleißgrad</span><br>
        <span class="val-main green-glow">{st.session_state.twin['wear']:.1f}<span class="unit">%</span></span>
    </div>
    """, unsafe_allow_html=True)

with col_chart:
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        
        # Bruchrisiko Chart
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', name="KI-Risiko", line=dict(color='#f85149', width=4)), row=1, col=1)
        # Sensorik Chart
        fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Last Nm", line=dict(color='#d29922', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['c'], y=df['amp']*1000, name="Vibration", line=dict(color='#3fb950', width=2)), row=2, col=1)
        
        fig.update_layout(height=580, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("System bereit. Drücken Sie Start für die Echtzeit-Telemetrie.")

with col_log:
    st.markdown('<p class="val-title">Machine Analytics Stream</p>', unsafe_allow_html=True)
    log_html = "".join([f"<div style='margin-bottom:5px;'>{l}</div>" for l in st.session_state.twin['logs'][:50]])
    st.markdown(f'<div class="terminal">{log_html}</div>', unsafe_allow_html=True)

# Footer Controls
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("▶ START / STOP SYSTEM", use_container_width=True): 
        st.session_state.twin['active'] = not st.session_state.twin['active']
with c2:
    if st.button("🔄 VOLLSTÄNDIGER RESET", use_container_width=True):
        st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
        st.rerun()

if st.session_state.twin['active']:
    time.sleep(0.05)
    st.rerun()
