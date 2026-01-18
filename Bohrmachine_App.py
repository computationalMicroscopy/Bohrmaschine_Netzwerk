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
    # Struktur: Alter & Material -> Zustand -> [Vibration, Strom, Temperatur, Akustik]
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('Material', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Akustik')
    ])
    
    # Priors
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) # Neu, Mittel, Alt
    cpd_m = TabularCPD('Material', 2, [[0.6], [0.4]])    # Weich, Hart
    
    # Zustand CPT (Kausalität zwischen Input und Hardware-Status)
    cpd_z = TabularCPD('Zustand', 3, [
            [0.98, 0.85, 0.60, 0.30, 0.05, 0.01], 
            [0.01, 0.10, 0.30, 0.40, 0.60, 0.20], 
            [0.01, 0.05, 0.10, 0.30, 0.35, 0.79]], 
            evidence=['Alter', 'Material'], evidence_card=[3, 2])
    
    # Sensor CPTs (Likelihoods)
    # Vibration: [Normal, Kritisch]
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.3, 0.05], [noise_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    # Strom: [Normal, Hoch]
    cpd_s = TabularCPD('Strom', 2, [[0.95, 0.2, 0.4], [0.05, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    # Temperatur: [Normal, Heiß]
    cpd_t = TabularCPD('Temperatur', 2, [[0.99, 0.1, 0.3], [0.01, 0.9, 0.7]], evidence=['Zustand'], evidence_card=[3])
    # Akustik (Kreischen): [Leise, Laut]
    cpd_ak = TabularCPD('Akustik', 2, [[0.90, 0.4, 0.5], [0.10, 0.6, 0.5]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_m, cpd_z, cpd_v, cpd_s, cpd_t, cpd_ak)
    return model

# --- 3. INITIALISIERUNG ---
if 'history' not in st.session_state:
    st.session_state.update({'count': 0, 'history': [], 'is_run': False, 'fail_next': False})

with st.sidebar:
    st.header("⚙️ Sensor-Kalibrierung")
    n_v = st.slider("Vibrations-Empfindlichkeit", 0.0, 0.4, 0.05)
    n_c = st.slider("Strom-Grundlast", 0.0, 0.4, 0.02)
    n_t = st.slider("Thermische Trägheit", 0.0, 0.4, 0.01)
    st.divider()
    speed = st.select_slider("Prozess-Takt", [0.5, 0.2, 0.1, 0.01], 0.1)

bn = create_complex_bn(n_v, n_c, n_t)
inf = VariableElimination(bn)

# --- 4. DASHBOARD HEADER ---
st.title("🏭 AI Digital Twin: Heavy Duty Drill Press")
col_btn1, col_btn2, col_btn3, col_info = st.columns([1, 1, 1, 2])
with col_btn1:
    if st.button("🚀 SYSTEM START/STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with col_btn2:
    if st.button("🧹 CACHE CLEAR", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False})
        st.rerun()
with col_btn3:
    if st.button("⚠️ BRUCH ERZWINGEN", type="primary", use_container_width=True): st.session_state.fail_next = True

# --- 5. SIMULATION & LOGIK ---
if st.session_state.is_run:
    st.session_state.count += 1
    age = min(2, st.session_state.count // 50)
    mat = 1 if np.random.random() < 0.3 else 0 # 30% hartes Material
    
    # Kausale Logik für Simulation
    true_z = 2 if st.session_state.fail_next else (1 if (age >= 1 and np.random.random() > 0.8) else 0)
    st.session_state.fail_next = False
    
    # Sensor-Werte generieren (mit Noise)
    v_amp = np.random.normal(loc=(80 if true_z == 2 else 20), scale=10) # Amplitude in mm/s
    t_val = np.random.normal(loc=(70 if true_z == 1 else 30), scale=5)  # Temp in °C
    
    # Diskretisierung für das BN
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
else:
    ev, res = None, [1, 0, 0]

# --- 6. VISUALISIERUNG ---
st.write("---")
col_sensors, col_brain = st.columns([1, 2])

with col_sensors:
    st.subheader("📟 Sensor-Telemetrie")
    
    # Vibration Gauge Ersatz
    v_color = "#ef4444" if ev and ev['Vibration'] else "#3b82f6"
    st.markdown(f"""<div class="sensor-tile">
        <small>VIBRATIONS-AMPLITUDE</small><br>
        <b style="color:{v_color}; font-size:25px;">{st.session_state.history[-1]['v_amp']:.1f if ev else 0} mm/s</b>
    </div>""", unsafe_allow_html=True)
    
    # Temperatur Gauge Ersatz
    t_color = "#f97316" if ev and ev['Temperatur'] else "#10b981"
    st.markdown(f"""<div class="sensor-tile">
        <small>THERMISCHE LAST</small><br>
        <b style="color:{t_color}; font-size:25px;">{st.session_state.history[-1]['temp']:.1f if ev else 0} °C</b>
    </div>""", unsafe_allow_html=True)
    
    # Binäre Sensoren
    c_s1, c_s2 = st.columns(2)
    c_s1.checkbox("Akustik-Emi. (Kreischen)", value=bool(ev['Akustik'] if ev else 0), disabled=True)
    c_s2.checkbox("Strom-Peak", value=bool(ev['Strom'] if ev else 0), disabled=True)

with col_brain:
    st.subheader("🧠 Probabilistisches Gehirn")
    
    # Wahrscheinlichkeits-Cockpit
    b1, b2, b3 = st.columns(3)
    b1.metric("Intakt", f"{res[0]:.1%}")
    b2.metric("Stumpf", f"{res[1]:.1%}")
    b3.metric("Bruch", f"{res[2]:.1%}", delta=f"{res[2]*100:.1f}%", delta_color="inverse")
    
    # Dynamischer Graph
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['Bruch'], name="Bruch-Risiko", line=dict(color='#ef4444', width=4)))
        fig.add_trace(go.Scatter(x=df['t'], y=df['Stumpf'], name="Verschleiß", line=dict(color='#f59e0b', width=2)))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,1)',
            height=300, margin=dict(l=0,r=0,t=0,b=0), font=dict(color="white"),
            xaxis=dict(showgrid=False), yaxis=dict(range=[0,1.1])
        )
        st.plotly_chart(fig, use_container_width=True)

# --- 7. DAS FASZINATION-FEATURE: KAUSALER FINGERABDRUCK ---
st.write("---")
st.subheader("🔬 Kausale Attribution (Digital Twin Analysis)")
if ev:
    st.markdown('<div class="logic-card">', unsafe_allow_html=True)
    
    # Berechnung des "Haupt-Beweisstücks"
    # Wir schauen, welche Evidenz am weitesten vom Standard abweicht
    attributions = []
    if ev['Vibration']: attributions.append("⚠️ Die **Vibrationsamplitude** ist kritisch. Dies ist der stärkste Indikator für strukturelles Versagen.")
    if ev['Temperatur']: attributions.append("🔥 Die **Temperatur** steigt. Dies validiert die Hypothese eines stumpfen Bohrers.")
    if ev['Akustik']: attributions.append("🔉 **Akustische Muster** bestätigen mechanische Unregelmäßigkeiten.")
    
    if not attributions:
        st.write("✅ Alle Systeme im Nominalbereich. Die Diagnose stützt sich primär auf die statistische Lebensdauer (Alter).")
    else:
        for a in attributions: st.write(a)
    
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(speed)
    st.rerun()
