import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI & THEME SETUP ---
st.set_page_config(layout="wide", page_title="Industrial AI Lab v3", page_icon="🏗️")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .sensor-tile { 
        background-color: #161b22; border-radius: 10px; padding: 15px; 
        border: 1px solid #30363d; margin-bottom: 10px;
    }
    .logic-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 20px; border-radius: 15px; border: 1px solid #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KOMPLEXES BAYESSCHES NETZWERK ---
@st.cache_resource
def create_complex_bn(noise_v, noise_c, noise_t):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('Material', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Akustik')
    ])
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_m = TabularCPD('Material', 2, [[0.6], [0.4]])    
    cpd_z = TabularCPD('Zustand', 3, [
            [0.98, 0.85, 0.60, 0.30, 0.05, 0.01], 
            [0.01, 0.10, 0.30, 0.40, 0.60, 0.20], 
            [0.01, 0.05, 0.10, 0.30, 0.35, 0.79]], 
            evidence=['Alter', 'Material'], evidence_card=[3, 2])
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.3, 0.05], [noise_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[0.95, 0.2, 0.4], [0.05, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[0.99, 0.1, 0.3], [0.01, 0.9, 0.7]], evidence=['Zustand'], evidence_card=[3])
    cpd_ak = TabularCPD('Akustik', 2, [[0.90, 0.4, 0.5], [0.10, 0.6, 0.5]], evidence=['Zustand'], evidence_card=[3])
    model.add_cpds(cpd_a, cpd_m, cpd_z, cpd_v, cpd_s, cpd_t, cpd_ak)
    return model

# --- 3. INITIALISIERUNG ---
# Einheitliche Benennung aller Keys
if 'history' not in st.session_state:
    st.session_state.update({
        'count': 0, 
        'history': [], 
        'is_run': False, 
        'manual_fail': False, 
        'emergency_stop': False
    })

with st.sidebar:
    st.header("⚙️ Sensor-Kalibrierung")
    n_v = st.slider("Vibrations-Empfindlichkeit", 0.0, 0.4, 0.05)
    n_c = st.slider("Strom-Grundlast", 0.0, 0.4, 0.02)
    n_t = st.slider("Thermische Trägheit", 0.0, 0.4, 0.01)
    st.divider()
    speed = st.select_slider("Prozess-Takt", [0.5, 0.2, 0.1, 0.01], 0.1)
    auto_stop_active = st.checkbox("Automatischer Not-Halt bei >90%", value=True)

bn = create_complex_bn(n_v, n_c, n_t)
inf = VariableElimination(bn)

# --- 4. DASHBOARD HEADER ---
st.title("🏭 AI Digital Twin: Heavy Duty Drill Press")
col_btn1, col_btn2, col_btn3, col_info = st.columns([1, 1, 1, 2])
with col_btn1:
    if st.button("🚀 SYSTEM START/STOP", use_container_width=True): 
        st.session_state.is_run = not st.session_state.is_run
        st.session_state.emergency_stop = False
with col_btn2:
    if st.button("🧹 CACHE CLEAR", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'emergency_stop': False, 'manual_fail': False})
        st.rerun()
with col_btn3:
    if st.button("⚠️ BRUCH ERZWINGEN", type="primary", use_container_width=True): 
        st.session_state.manual_fail = True

# --- 5. SIMULATION & LOGIK ---
if st.session_state.is_run:
    st.session_state.count += 1
    age = min(2, st.session_state.count // 50)
    mat = 1 if np.random.random() < 0.3 else 0 
    
    # Kausale Simulation: Nutzt jetzt korrekt manual_fail
    if st.session_state.manual_fail:
        true_z = 2
        st.session_state.manual_fail = False
    else:
        # Natürlicher Verschleiß-Zufall
        true_z = 1 if (age >= 1 and np.random.random() > 0.8) else 0
    
    v_amp = np.random.normal(loc=(80 if true_z == 2 else 20), scale=10) 
    t_val = np.random.normal(loc=(70 if true_z == 1 else 30), scale=5)  
    
    ev = {
        'Vibration': 1 if v_amp > 50 else 0,
        'Strom': 1 if (true_z == 1 or np.random.random() < n_c) else 0,
        'Temperatur': 1 if t_val > 55 else 0,
        'Akustik': 1 if (true_z > 0 and np.random.random() > 0.4) else 0,
        'Material': mat,
        'Alter': age
    }
    res = inf.query(['Zustand'], evidence=ev).values
    st.session_state.history.append({'t': st.session_state.count, 'Intakt': res[0], 'Stumpf': res[1], 'Bruch': res[2], 'v_amp': v_amp, 'temp': t_val})
    
    if auto_stop_active and res[2] > 0.9:
        st.session_state.is_run = False
        st.session_state.emergency_stop = True
else:
    # Default Werte wenn Simulation steht
    res = [1, 0, 0]
    ev = None

# --- 6. VISUALISIERUNG ---
if st.session_state.emergency_stop:
    st.error("🚨 NOT-HALT AUSGELÖST: Kritische Bruchwahrscheinlichkeit detektiert!")

st.write("---")
col_sensors, col_brain = st.columns([1, 2])

with col_sensors:
    st.subheader("📟 Sensor-Telemetrie")
    last_v = st.session_state.history[-1]['v_amp'] if st.session_state.history else 0
    last_t = st.session_state.history[-1]['temp'] if st.session_state.history else 0
    
    v_color = "#ef4444" if last_v > 50 else "#3b82f6"
    st.markdown(f"""<div class="sensor-tile">
        <small>VIBRATIONS-AMPLITUDE</small><br>
        <b style="color:{v_color}; font-size:25px;">{last_v:.1f} mm/s</b>
    </div>""", unsafe_allow_html=True)
    
    t_color = "#f97316" if last_t > 55 else "#10b981"
    st.markdown(f"""<div class="sensor-tile">
        <small>THERMISCHE LAST</small><br>
        <b style="color:{t_color}; font-size:25px;">{last_t:.1f} °C</b>
    </div>""", unsafe_allow_html=True)
    
    c_s1, c_s2 = st.columns(2)
    akustik_val = bool(ev['Akustik'] if ev else 0)
    strom_val = bool(ev['Strom'] if ev else 0)
    c_s1.checkbox("Akustik (Kreischen)", value=akustik_val, disabled=True)
    c_s2.checkbox("Strom-Peak", value=strom_val, disabled=True)

with col_brain:
    st.subheader("🧠 Probabilistisches Gehirn")
    b1, b2, b3 = st.columns(3)
    b1.metric("Intakt", f"{res[0]:.1%}")
    b2.metric("Stumpf", f"{res[1]:.1%}")
    b3.metric("Bruch", f"{res[2]:.1%}", delta=f"{res[2]*100:.1f}%", delta_color="inverse")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['Bruch'], name="Bruch-Risiko", line=dict(color='#ef4444', width=4)))
        fig.add_trace(go.Scatter(x=df['t'], y=df['Stumpf'], name="Verschleiß", line=dict(color='#f59e0b', width=2)))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,1)', height=300, margin=dict(l=0,r=0,t=0,b=0), font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.subheader("🔬 Kausale Attribution (Digital Twin Analysis)")
if ev:
    st.markdown('<div class="logic-card">', unsafe_allow_html=True)
    if ev['Vibration']: st.write("⚠️ Kritische **Vibrationsamplitude** erkannt.")
    if ev['Temperatur']: st.write("🔥 **Thermische Last** erhöht (Reibung).")
    if ev['Akustik']: st.write("🔉 **Akustik-Sensor** meldet Unregelmäßigkeiten.")
    if not any([ev['Vibration'], ev['Temperatur'], ev['Akustik']]): st.write("✅ Alle Messwerte nominal.")
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(speed)
    st.rerun()
