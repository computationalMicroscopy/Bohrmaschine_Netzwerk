import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI & THEME SETUP ---
st.set_page_config(layout="wide", page_title="Industrial AI Lab v4", page_icon="🏗️")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .sensor-tile { 
        background-color: #161b22; border-radius: 10px; padding: 12px; 
        border: 1px solid #30363d; margin-bottom: 8px;
    }
    .log-container {
        height: 300px; overflow-y: scroll; background-color: #0d1117;
        border: 1px solid #30363d; border-radius: 10px; padding: 15px;
        font-family: 'Courier New', Courier, monospace; font-size: 0.85rem;
    }
    .log-entry { margin-bottom: 5px; border-bottom: 1px solid #21262d; padding-bottom: 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ERWEITERTES BAYESSCHES NETZWERK (7 Sensoren) ---
@st.cache_resource
def create_complex_bn(noise_v, noise_c, noise_t):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('Material', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Akustik'),
        ('Zustand', 'Drehmoment'), ('Zustand', 'Vorschub')
    ])
    
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_m = TabularCPD('Material', 2, [[0.6], [0.4]])    
    cpd_z = TabularCPD('Zustand', 3, [
            [0.98, 0.85, 0.60, 0.30, 0.05, 0.01], 
            [0.01, 0.10, 0.30, 0.40, 0.60, 0.20], 
            [0.01, 0.05, 0.10, 0.30, 0.35, 0.79]], 
            evidence=['Alter', 'Material'], evidence_card=[3, 2])
    
    # Sensoren Likelihoods
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.3, 0.05], [noise_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[0.95, 0.2, 0.4], [0.05, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[0.99, 0.1, 0.3], [0.01, 0.9, 0.7]], evidence=['Zustand'], evidence_card=[3])
    cpd_ak = TabularCPD('Akustik', 2, [[0.90, 0.4, 0.5], [0.10, 0.6, 0.5]], evidence=['Zustand'], evidence_card=[3])
    cpd_dr = TabularCPD('Drehmoment', 2, [[0.95, 0.3, 0.6], [0.05, 0.7, 0.4]], evidence=['Zustand'], evidence_card=[3])
    cpd_vo = TabularCPD('Vorschub', 2, [[0.98, 0.4, 0.1], [0.02, 0.6, 0.9]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_m, cpd_z, cpd_v, cpd_s, cpd_t, cpd_ak, cpd_dr, cpd_vo)
    return model

# --- 3. INITIALISIERUNG ---
if 'history' not in st.session_state:
    st.session_state.update({
        'count': 0, 'history': [], 'is_run': False, 
        'manual_fail': False, 'emergency_stop': False, 'event_logs': []
    })

with st.sidebar:
    st.header("⚙️ Konfiguration")
    n_v = st.slider("Vibrations-Noise", 0.0, 0.4, 0.05)
    n_c = st.slider("Strom-Noise", 0.0, 0.4, 0.02)
    n_t = st.slider("Temperatur-Noise", 0.0, 0.4, 0.01)
    st.divider()
    speed = st.select_slider("Takt", [1.0, 0.5, 0.2, 0.1, 0.01], 0.1)
    auto_stop_active = st.checkbox("Not-Halt aktiv", value=True)

bn = create_complex_bn(n_v, n_c, n_t)
inf = VariableElimination(bn)

# --- 4. HEADER ---
st.title("🏭 AI Digital Twin Drill Station")
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
with c1:
    if st.button("🚀 START / STOP", use_container_width=True): 
        st.session_state.is_run = not st.session_state.is_run
        st.session_state.emergency_stop = False
with c2:
    if st.button("🧹 RESET", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'emergency_stop': False, 'manual_fail': False, 'event_logs': []})
        st.rerun()
with c3:
    if st.button("⚠️ FEHLER", type="primary", use_container_width=True): st.session_state.manual_fail = True
with c4:
    st.subheader(f"Bohrvorgänge: {st.session_state.count}")

# --- 5. SIMULATION ---
if st.session_state.is_run:
    st.session_state.count += 1
    age = min(2, st.session_state.count // 50)
    mat = 1 if np.random.random() < 0.3 else 0 
    
    if st.session_state.manual_fail:
        true_z = 2
        st.session_state.manual_fail = False
    else:
        true_z = 1 if (age >= 1 and np.random.random() > 0.85) else 0
    
    # Physikalische Werte
    v_amp = np.random.normal(loc=(85 if true_z == 2 else 15), scale=8) 
    t_val = np.random.normal(loc=(75 if true_z == 1 else 28), scale=4)
    torque = np.random.normal(loc=(60 if true_z > 0 else 30), scale=5)
    
    ev = {
        'Vibration': 1 if v_amp > 50 else 0, 'Strom': 1 if (true_z == 1 or np.random.random() < n_c) else 0,
        'Temperatur': 1 if t_val > 55 else 0, 'Akustik': 1 if (true_z > 0 and np.random.random() > 0.4) else 0,
        'Drehmoment': 1 if torque > 45 else 0, 'Vorschub': 1 if (true_z == 2) else 0,
        'Material': mat, 'Alter': age
    }
    res = inf.query(['Zustand'], evidence=ev).values
    
    # Log-Eintrag erstellen
    timestamp = time.strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] ID {st.session_state.count}: "
    if res[2] > 0.5: log_msg += "❌ KRITISCH: Bruchgefahr!"
    elif res[1] > 0.5: log_msg += "⚠️ WARNUNG: Verschleiß hoch."
    else: log_msg += "✅ Normalbetrieb."
    st.session_state.event_logs.insert(0, log_msg) # Neueste oben
    
    st.session_state.history.append({'t': st.session_state.count, 'Intakt': res[0], 'Stumpf': res[1], 'Bruch': res[2], 'v_amp': v_amp, 'temp': t_val, 'torque': torque})
    
    if auto_stop_active and res[2] > 0.9:
        st.session_state.is_run = False
        st.session_state.emergency_stop = True
else:
    res = [1, 0, 0]
    ev = None

# --- 6. DISPLAY ---
if st.session_state.emergency_stop:
    st.error("🚨 NOT-HALT: Bruchwahrscheinlichkeit > 90%!")

st.write("---")
col_tele, col_graph, col_log = st.columns([1, 2, 1.2])

with col_tele:
    st.subheader("📡 Telemetrie")
    l_v = st.session_state.history[-1]['v_amp'] if st.session_state.history else 0
    l_t = st.session_state.history[-1]['temp'] if st.session_state.history else 0
    l_d = st.session_state.history[-1]['torque'] if st.session_state.history else 0
    
    st.markdown(f'<div class="sensor-tile"><small>VIBRATION</small><br><b style="color:{"#ef4444" if l_v > 50 else "#3b82f6"}">{l_v:.1f} mm/s</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile"><small>TEMPERATUR</small><br><b style="color:{"#f97316" if l_t > 55 else "#10b981"}">{l_t:.1f} °C</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile"><small>DREHMOMENT</small><br><b>{l_d:.1f} Nm</b></div>', unsafe_allow_html=True)
    
    st.caption("Binäre Status-Bits")
    st.checkbox("Akustik-Alarm", value=bool(ev['Akustik'] if ev else 0), disabled=True)
    st.checkbox("Vorschub-Stopp", value=bool(ev['Vorschub'] if ev else 0), disabled=True)

with col_graph:
    st.subheader("🧠 KI-Analyse & Historie")
    b1, b2, b3 = st.columns(3)
    b1.metric("Intakt", f"{res[0]:.1%}")
    b2.metric("Stumpf", f"{res[1]:.1%}")
    b3.metric("Bruch", f"{res[2]:.1%}", delta=f"{res[2]*100:.1f}%", delta_color="inverse")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['Bruch'], name="Bruch", line=dict(color='#ef4444', width=3)))
        fig.add_trace(go.Scatter(x=df['t'], y=df['Stumpf'], name="Verschleiß", line=dict(color='#f59e0b', width=2)))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,1)', height=250, margin=dict(l=0,r=0,t=0,b=0), font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

with col_log:
    st.subheader("📜 Digital Twin Log")
    log_html = "".join([f'<div class="log-entry">{log}</div>' for log in st.session_state.event_logs])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

# Kausale Erläuterung (feststehend unter Log)
if ev:
    with st.expander("💡 Aktuelle Kausal-Analyse", expanded=True):
        if ev['Vibration'] and ev['Akustik']: st.write("Starke Korrelation zwischen Vibration und Akustik -> **Struktureller Defekt**.")
        if ev['Temperatur'] and ev['Drehmoment']: st.write("Reibungswärme + Drehmoment-Anstieg -> **Hoher Verschleiß**.")
        if ev['Vorschub']: st.write("Vorschub blockiert -> **Bohrer klemmt/gebrochen**.")

if st.session_state.is_run:
    time.sleep(speed)
    st.rerun()
