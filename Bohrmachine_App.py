import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. Konfiguration & UI Setup ---
st.set_page_config(layout="wide", page_title="ML Schulung: Bayessche Netze")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Sidebar: Parameter ---
st.sidebar.header("🛠️ Maschinen-Konfiguration")

st.sidebar.subheader("Sensor-Präzision")
noise_level = st.sidebar.slider("Sensor-Rauschen (Fehlalarm-Rate)", 0.0, 0.5, 0.1)

st.sidebar.subheader("Material-Einfluss")
hardness_impact = st.sidebar.slider("Material-Härte (Verschleißfaktor)", 1.0, 5.0, 2.0)

st.sidebar.subheader("Simulation")
# FIX: value=0.3 ist jetzt in options enthalten
sim_speed = st.sidebar.select_slider(
    "Simulations-Pause (Sekunden)", 
    options=[1.0, 0.5, 0.3, 0.1, 0.01], 
    value=0.3
)

# --- 3. Netzwerk-Logik ---
@st.cache_resource
def create_interactive_bn(noise, impact):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('Material', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom')
    ])

    cpd_alter = TabularCPD(variable='Alter', variable_card=3, values=[[0.7], [0.2], [0.1]])
    cpd_material = TabularCPD(variable='Material', variable_card=2, values=[[0.5], [0.5]])

    cpd_zustand = TabularCPD(
        variable='Zustand', variable_card=3,
        values=[
            [0.99, 0.90, 0.70, 0.40, 0.10, 0.01], # Intakt
            [0.01, 0.09, 0.25, 0.40, 0.60, 0.30], # Stumpf
            [0.00, 0.01, 0.05, 0.20, 0.30, 0.69]  # Gebrochen
        ],
        evidence=['Alter', 'Material'], evidence_card=[3, 2]
    )

    cpd_vibration = TabularCPD(
        variable='Vibration', variable_card=2,
        values=[[1-noise, 0.4, 0.1], [noise, 0.6, 0.9]],
        evidence=['Zustand'], evidence_card=[3]
    )
    
    cpd_strom = TabularCPD(
        variable='Strom', variable_card=2,
        values=[[0.95, 0.2, 0.5], [0.05, 0.8, 0.5]],
        evidence=['Zustand'], evidence_card=[3]
    )

    model.add_cpds(cpd_alter, cpd_material, cpd_zustand, cpd_vibration, cpd_strom)
    return model

# --- 4. Session State ---
if 'history' not in st.session_state:
    st.session_state.update({
        'drilling_count': 0, 
        'alter_state': 0, 
        'history': [], 
        'is_running': False, 
        'manual_fail': False
    })

bn_model = create_interactive_bn(noise_level, hardness_impact)
inference = VariableElimination(bn_model)

# --- 5. Haupt-Layout ---
st.title("🎓 Interactive Machine Learning Lab")
st.write("Thema: **Condition Monitoring mit Bayesschen Netzwerken**")

col_left, col_mid, col_right = st.columns([1, 1, 1.5])

with col_left:
    st.subheader("📡 Live-Sensordaten")
    manual_mode = st.toggle("Manueller Sensor-Override")
    
    if manual_mode:
        st.session_state.is_running = False
        m_vib = st.radio("Vibration", ["Niedrig", "Hoch"])
        m_str = st.radio("Stromaufnahme", ["Normal", "Hoch"])
        m_mat = st.selectbox("Material", ["Weich", "Hart"])
        m_alt = st.select_slider("Bohrer-Alter", ["Neu", "Mittel", "Alt"])
        
        ev = {
            'Vibration': 0 if m_vib == "Niedrig" else 1,
            'Strom': 0 if m_str == "Normal" else 1,
            'Material': 0 if m_mat == "Weich" else 1,
            'Alter': ["Neu", "Mittel", "Alt"].index(m_alt)
        }
    else:
        if st.button("Simulation Start / Stop"):
            st.session_state.is_running = not st.session_state.is_running
        
        if st.button("🚨 FEHLER INJIZIEREN"):
            st.session_state.manual_fail = True
        
        if st.session_state.is_running:
            st.session_state.drilling_count += 1
            st.session_state.alter_state = min(2, st.session_state.drilling_count // 50)
            
            true_z = 2 if st.session_state.manual_fail else (1 if np.random.random() > 0.85 else 0)
            st.session_state.manual_fail = False 
            
            v_val = 1 if (true_z > 0 or np.random.random() < noise_level) else 0
            s_val = 1 if (true_z == 1 or np.random.random() < 0.1) else 0
            mat_val = np.random.choice([0, 1])
            
            ev = {'Vibration': v_val, 'Strom': s_val, 'Material': mat_val, 'Alter': st.session_state.alter_state}
            st.write(f"Vorgang: {st.session_state.drilling_count}")
        else:
            ev = None

with col_mid:
    st.subheader("🧠 KI-Diagnose")
    if ev:
        res = inference.query(variables=['Zustand'], evidence=ev).values
        st.metric("Sicherheit: Intakt", f"{res[0]:.1%}")
        st.metric("Sicherheit: Stumpf", f"{res[1]:.1%}")
        st.metric("Sicherheit: Gebrochen", f"{res[2]:.1%}", delta=f"{res[2]:.1%}" if res[2] > 0.5 else None, delta_color="inverse")
        
        if st.session_state.is_running:
            st.session_state.history.append({'x': st.session_state.drilling_count, 'i': res[0], 's': res[1], 'g': res[2]})
    else:
        st.info("Simulation starten.")

with col_right:
    st.subheader("📈 Wahrscheinlichkeitsverlauf")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['x'], y=df['i'], name="Intakt", fill='tozeroy', line_color='green'))
        fig.add_trace(go.Scatter(x=df['x'], y=df['s'], name="Stumpf", fill='tonexty', line_color='orange'))
        fig.add_trace(go.Scatter(x=df['x'], y=df['g'], name="Gebrochen", fill='tonexty', line_color='red'))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0), yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

if st.session_state.is_running:
    time.sleep(sim_speed)
    st.rerun()
