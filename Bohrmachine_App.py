import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. Seiteneinstellungen & Styling ---
st.set_page_config(layout="wide", page_title="AI Predictive Maintenance Lab", page_icon="⚙️")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { border: 1px solid #d1d8e0; padding: 15px; border-radius: 10px; background: white; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .sensor-card { background: #ffffff; padding: 20px; border-radius: 15px; border-left: 5px solid #3498db; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Sidebar: Fortgeschrittene Parameter ---
st.sidebar.title("🛠️ Labor-Konfiguration")

with st.sidebar.expander("📈 BN-Modell Parameter", expanded=True):
    prob_hard_material = st.slider("Häufigkeit Hartes Material", 0.1, 0.9, 0.5, 0.05)
    wear_speed = st.slider("Verschleiß-Beschleuniger", 1.0, 5.0, 2.0)

with st.sidebar.expander("📡 Sensor-Charakteristik"):
    vib_noise = st.slider("Rauschen Vibration", 0.0, 0.4, 0.1)
    current_noise = st.slider("Rauschen Stromaufnahme", 0.0, 0.4, 0.05)

with st.sidebar.expander("🚨 Alarm-Management"):
    threshold_warning = st.slider("Warnung-Schwellenwert", 0.3, 0.9, 0.6)
    sim_speed = st.select_slider("Simulations-Takt (Sek.)", options=[1.0, 0.5, 0.3, 0.1, 0.01], value=0.3)

# --- 3. Bayessches Netzwerk Kern ---
@st.cache_resource
def create_advanced_bn(noise_v, noise_c, hard_prob, wear):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('Material', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom')
    ])

    cpd_alter = TabularCPD(variable='Alter', variable_card=3, values=[[0.7], [0.2], [0.1]])
    cpd_material = TabularCPD(variable='Material', variable_card=2, values=[[1-hard_prob], [hard_prob]])

    # Dynamische Zustands-CPT basierend auf Verschleiß-Parameter
    cpd_zustand = TabularCPD(
        variable='Zustand', variable_card=3,
        values=[
            [0.99, 0.85, 0.70, 0.30, 0.05, 0.01], # Intakt
            [0.01, 0.10, 0.20, 0.40, 0.50, 0.30], # Stumpf
            [0.00, 0.05, 0.10, 0.30, 0.45, 0.69]  # Gebrochen
        ],
        evidence=['Alter', 'Material'], evidence_card=[3, 2]
    )

    cpd_vibration = TabularCPD(
        variable='Vibration', variable_card=2,
        values=[[1-noise_v, 0.4, 0.1], [noise_v, 0.6, 0.9]],
        evidence=['Zustand'], evidence_card=[3]
    )
    
    cpd_strom = TabularCPD(
        variable='Strom', variable_card=2,
        values=[[1-noise_c, 0.2, 0.5], [noise_c, 0.8, 0.5]],
        evidence=['Zustand'], evidence_card=[3]
    )

    model.add_cpds(cpd_alter, cpd_material, cpd_zustand, cpd_vibration, cpd_strom)
    return model

# --- 4. Session Management ---
if 'history' not in st.session_state:
    st.session_state.update({'drilling_count': 0, 'history': [], 'is_running': False, 'manual_fail': False})

bn_model = create_advanced_bn(vib_noise, current_noise, prob_hard_material, wear_speed)
inference = VariableElimination(bn_model)

# --- 5. Haupt-Interface ---
st.title("🛡️ AI Reliability & Maintenance Lab")
st.markdown("### Probabilistische Zustandsüberwachung einer Standbohrmaschine")

# Obere Reihe: Status-Karten
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# Simulation Logik
if st.session_state.is_running:
    st.session_state.drilling_count += 1
    current_age = min(2, st.session_state.drilling_count // 40)
    
    # Sampling (vereinfacht für UI-Reaktion)
    true_z = 2 if st.session_state.manual_fail else (1 if np.random.random() > 0.88 else 0)
    st.session_state.manual_fail = False
    
    v_val = 1 if (true_z > 0 or np.random.random() < vib_noise) else 0
    s_val = 1 if (true_z == 1 or np.random.random() < current_noise) else 0
    mat_val = 1 if np.random.random() < prob_hard_material else 0
    
    ev = {'Vibration': v_val, 'Strom': s_val, 'Material': mat_val, 'Alter': current_age}
else:
    ev = None
    current_age = 0

# UI Komponenten
with m_col1:
    st.metric("Bohrvorgänge", st.session_state.drilling_count)
with m_col2:
    age_labels = ["Neu", "Mittel", "Alt"]
    st.metric("Bohrer-Alter", age_labels[current_age])
with m_col3:
    status_text = "Simulation Inaktiv"
    if ev:
        status_text = "Hart" if ev['Material'] == 1 else "Weich"
    st.metric("Aktuelles Material", status_text)
with m_col4:
    st.metric("Sensor-Status", "Online" if st.session_state.is_running else "Standby")

st.write("---")

# Mittlere Reihe: Analyse & Lab-Steuerung
l_col, r_col = st.columns([1, 2])

with l_col:
    st.subheader("🕹️ Steuerung")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start / Stop", use_container_width=True):
            st.session_state.is_running = not st.session_state.is_running
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.drilling_count = 0
            st.session_state.history = []
            st.rerun()
    
    if st.button("💥 FEHLER INJIZIEREN (Bohrerbruch)", type="primary", use_container_width=True):
        st.session_state.manual_fail = True

    st.write("---")
    st.subheader("🔍 Aktuelle Inferenz")
    if ev:
        res = inference.query(variables=['Zustand'], evidence=ev).values
        
        # Diagnose-Anzeige mit Fortschrittsbalken
        for i, label in enumerate(["Intakt", "Stumpf", "Gebrochen"]):
            cols = st.columns([1, 4])
            cols[0].write(label)
            cols[1].progress(float(res[i]))
        
        if res[2] > threshold_warning:
            st.error(f"KRITISCH: Wahrscheinlichkeit für Bruch bei {res[2]:.1%}")
        elif res[1] > threshold_warning:
            st.warning(f"WARNUNG: Verschleiß erkannt ({res[1]:.1%})")
            
        if st.session_state.is_running:
            st.session_state.history.append({'t': st.session_state.drilling_count, 'i': res[0], 's': res[1], 'g': res[2]})
    else:
        st.info("Starten Sie die Simulation für Live-Diagnose.")

with r_col:
    st.subheader("📊 Wahrscheinlichkeits-Historie")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['i'], name="Intakt", fill='tozeroy', line_color='#2ecc71'))
        fig.add_trace(go.Scatter(x=df['t'], y=df['s'], name="Stumpf", fill='tonexty', line_color='#f1c40f'))
        fig.add_trace(go.Scatter(x=df['t'], y=df['g'], name="Gebrochen", fill='tonexty', line_color='#e74c3c'))
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            xaxis_title="Zeitverlauf (Vorgänge)",
            yaxis_title="Konfidenz",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Drill_Press_BW_2015-11-09_14-38-00.jpg/640px-Drill_Press_BW_2015-11-09_14-38-00.jpg", caption="Bereit für Simulation")

# Automatischer Rerun
if st.session_state.is_running:
    time.sleep(sim_speed)
    st.rerun()
