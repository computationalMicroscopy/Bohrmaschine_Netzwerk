import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. INDUSTRIAL DESIGN ---
st.set_page_config(layout="wide", page_title="AI Precision Twin v21 - Spectral", page_icon="🧬")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #c9d1d9; }
    .metric-box { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 4px; padding: 10px; text-align: center;
    }
    .status-stable { color: #3fb950; font-weight: bold; }
    .status-warn { color: #d29922; font-weight: bold; }
    .status-crit { color: #f85149; font-weight: bold; }
    .log-terminal { font-family: 'Courier New', monospace; font-size: 0.75rem; height: 300px; overflow-y: auto; background: #010409; color: #3fb950; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PHYSIK-KONSTANTEN & MATERIAL ---
MATERIALIEN = {
    "Stahl 42CrMo4": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "freq_base": 120},
    "Titan Gr. 5": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.2, "freq_base": 95},
    "Inconel 718": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.4, "freq_base": 80}
}

# --- 3. SESSION STATE ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'bearing_damage': 0.0, 'rng': np.random.RandomState(42)
    }

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🧬 System-Analyse")
    mat_name = st.selectbox("Material", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    
    with st.expander("Prozess-Stellgrößen", expanded=True):
        vc = st.slider("vc [m/min]", 20, 400, 140)
        f = st.slider("f [mm/U]", 0.05, 0.6, 0.15)
        d_tool = st.number_input("Bohrer-Ø [mm]", 1.0, 40.0, 10.0)
        cooling = st.toggle("Hochdruck-Kühlung", value=True)

    with st.expander("Maschinenzustand"):
        sim_speed = st.select_slider("Taktung", options=[500, 100, 50, 10, 0], value=50)
        bearing_fail = st.slider("Lager-Verschleiß", 0.0, 1.0, 0.0)
        noise_level = st.slider("Signalrauschen", 0.0, 0.5, 0.05)

# --- 5. SPECTRAL ENGINE (FFT SIMULATION) ---
def generate_spectral_data(wear, bearing_dmg, base_freq, noise):
    x = np.linspace(0, 500, 200) # Frequenzbereich 0-500 Hz
    # Grundfrequenz des Prozesses
    y = 5 * np.exp(-((x - base_freq)**2) / 10)
    # Verschleiß-Harmonische (steigt bei wear > 70)
    if wear > 50:
        y += (wear/20) * np.exp(-((x - base_freq*2)**2) / 15)
    # Lagerschaden (spezifischer Peak bei 340Hz)
    if bearing_dmg > 0:
        y += (bearing_dmg * 15) * np.exp(-((x - 340)**2) / 5)
    # Grundrauschen
    y += np.random.normal(0, noise * 2, 200)
    return x, np.clip(y, 0, 20)

# --- 6. CORE LOGIC ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 1
    
    # Mechanische Last
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d_tool/2)
    mc = (fc * d_tool) / 2000
    
    # Verschleiß-Entwicklung
    s['wear'] += (mat['wear_rate'] * (vc**1.5) * f) / (12000 if cooling else 500)
    
    # Thermik
    target_t = 22 + (s['wear'] * 1.3) + (vc * 0.15) + (0 if cooling else 200)
    s['t_current'] += (target_t - s['t_current']) * 0.1
    
    # Risiko-Inferenz (vereinfacht für v21 Performance)
    risk = min(0.99, (s['wear']**2.5 / 25000) + (mc / (d_tool*3)) + (0.3 if not cooling else 0))
    
    if risk > 0.97 or s['wear'] > 140:
        s['broken'] = True
        s['active'] = False
        
    s['history'].append({'c': s['cycle'], 'r': risk, 'w': s['wear'], 't': s['t_current'], 'mc': mc})
    s['logs'].insert(0, f">> SYNC CYC {s['cycle']}: FFT ANALYSED - SPECTRUM STABLE")

# --- 7. BEDIENOBERFLÄCHE ---
st.title("🔩 AI Precision Twin v21: Spectral Analytics")

# Top Metrics
t1, t2, t3, t4 = st.columns(4)
last = st.session_state.twin['history'][-1] if st.session_state.twin['history'] else {'c':0,'r':0,'w':0,'t':22,'mc':0}

with t1: st.markdown(f'<div class="metric-box"><small>ZYKLUS</small><br><span style="font-size:2rem; font-weight:bold;">{last["c"]}</span></div>', unsafe_allow_html=True)
with t2: st.markdown(f'<div class="metric-box"><small>DREHMOMENT</small><br><span style="font-size:2rem; font-weight:bold; color:#58a6ff;">{last["mc"]:.2f} Nm</span></div>', unsafe_allow_html=True)
with t3: st.markdown(f'<div class="metric-box"><small>VERSCHLEISS</small><br><span style="font-size:2rem; font-weight:bold; color:#e3b341;">{last["w"]:.1f} %</span></div>', unsafe_allow_html=True)
with t4:
    status = "CRITICAL" if last['r'] > 0.8 else ("WARNING" if last['r'] > 0.5 else "STABLE")
    st.markdown(f'<div class="metric-box"><small>PROZESS-STATUS</small><br><span class="status-{status.lower()}">{status}</span></div>', unsafe_allow_html=True)

st.write("")

# FFT & Oszilloskop
col_fft, col_trends = st.columns([2, 1])

with col_fft:
    st.subheader("📡 Echtzeit-Frequenzspektrum (FFT)")
    freq_x, amp_y = generate_spectral_data(st.session_state.twin['wear'], bearing_fail, mat['freq_base'], noise_level)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freq_x, y=amp_y, fill='tozeroy', line=dict(color='#3fb950', width=2), name="Spektrum"))
    # Markierung für Lagerschaden
    if bearing_fail > 0.5:
        fig_fft.add_annotation(x=340, y=15, text="LAGERSCHADEN!", showarrow=True, arrowhead=1, bgcolor="#f85149")
    fig_fft.update_layout(height=350, template="plotly_dark", margin=dict(l=20,r=20,t=20,b=20), xaxis_title="Frequenz (Hz)", yaxis_title="Amplitude (mm)")
    st.plotly_chart(fig_fft, use_container_width=True)

with col_trends:
    st.subheader("📈 Risiko-Trend")
    if st.session_state.twin['history']:
        df = pd.DataFrame(st.session_state.twin['history'])
        fig_trend = go.Figure(go.Scatter(x=df['c'], y=df['r']*100, fill='tozeroy', line=dict(color='#f85149')))
        fig_trend.update_layout(height=350, template="plotly_dark", margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

# Footer Controls & Log
c_l, c_r = st.columns([1, 1])
with c_l:
    if st.button("▶️ SYSTEM START / PAUSE", use_container_width=True): st.session_state.twin['active'] = not st.session_state.twin['active']
    if st.button("🔄 HARD RESET", use_container_width=True):
        st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'bearing_damage':0.0,'rng':np.random.RandomState(42)}
        st.rerun()
with c_r:
    st.markdown(f'<div class="log-terminal">{"".join([f"<div>{l}</div>" for l in st.session_state.twin["logs"][:15]])}</div>', unsafe_allow_html=True)

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
