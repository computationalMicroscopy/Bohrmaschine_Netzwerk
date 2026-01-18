import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI SETUP & THEME ---
st.set_page_config(layout="wide", page_title="AI Bohr-Labor 2.0", page_icon="💎")

# Custom CSS für ein Dark-Industrial Design
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .status-card { 
        background-color: #1a1c24; border-radius: 15px; padding: 20px; 
        border: 1px solid #30363d; text-align: center;
    }
    .feature-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIK: BAYESSIAN NETWORK ---
@st.cache_resource
def create_bn(noise_v, noise_c, hard_prob):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('Material', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom')
    ])
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]])
    cpd_m = TabularCPD('Material', 2, [[1-hard_prob], [hard_prob]])
    cpd_z = TabularCPD('Zustand', 3, [
            [0.99, 0.85, 0.70, 0.30, 0.05, 0.01], 
            [0.01, 0.10, 0.20, 0.40, 0.50, 0.30], 
            [0.00, 0.05, 0.10, 0.30, 0.45, 0.69]], 
            evidence=['Alter', 'Material'], evidence_card=[3, 2])
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.4, 0.1], [noise_v, 0.6, 0.9]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[1-noise_c, 0.2, 0.5], [noise_c, 0.8, 0.5]], evidence=['Zustand'], evidence_card=[3])
    model.add_cpds(cpd_a, cpd_m, cpd_z, cpd_v, cpd_s)
    return model

# --- 3. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.update({'count': 0, 'history': [], 'is_run': False, 'fail_next': False})

# --- 4. SIDEBAR PARAMETER ---
with st.sidebar:
    st.title("🛠️ Labor-Zentrale")
    st.subheader("Umwelt & Sensorik")
    h_prob = st.slider("Anteil Hartes Material", 0.0, 1.0, 0.3)
    n_v = st.slider("Vibrations-Rauschen", 0.0, 0.5, 0.05)
    n_c = st.slider("Strom-Rauschen", 0.0, 0.5, 0.02)
    st.divider()
    speed = st.select_slider("Taktung", [1.0, 0.5, 0.1, 0.01], 0.1)

bn = create_bn(n_v, n_c, h_prob)
inf = VariableElimination(bn)

# --- 5. MAIN UI ---
st.title("⚙️ AI Bohr-Labor: Condition Monitoring")

# Obere Kontrollleiste
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("▶️ START / STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with c2:
    if st.button("🔄 SYSTEM RESET", use_container_width=True): 
        st.session_state.update({'count':0, 'history':[], 'is_run':False})
        st.rerun()
with c3:
    if st.button("💥 FEHLER INJIZIEREN", type="primary", use_container_width=True): st.session_state.fail_next = True
with c4:
    st.markdown(f"**Vorgänge:** {st.session_state.count}")

# Simulation
if st.session_state.is_run:
    st.session_state.count += 1
    age = min(2, st.session_state.count // 40)
    mat = 1 if np.random.random() < h_prob else 0
    true_z = 2 if st.session_state.fail_next else (1 if (age==2 and np.random.random()>0.7) else 0)
    st.session_state.fail_next = False
    v_s = 1 if (true_z > 0 or np.random.random() < n_v) else 0
    c_s = 1 if (true_z == 1 or np.random.random() < n_c) else 0
    ev = {'Vibration': v_s, 'Strom': c_s, 'Material': mat, 'Alter': age}
    res = inf.query(['Zustand'], evidence=ev).values
    st.session_state.history.append({'t': st.session_state.count, 'Intakt': res[0], 'Stumpf': res[1], 'Bruch': res[2]})
else:
    ev, res = None, [0, 0, 0]

# Dashboard-Bereich
st.write("---")
left, right = st.columns([1, 1.5])

with left:
    st.subheader("🎯 Live-Zustandsanalyse")
    # Gauges für die Zustände
    for i, (name, color) in enumerate(zip(["Intakt", "Stumpf", "Bruch"], ["#22c55e", "#eab308", "#ef4444"])):
        val = res[i]
        st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex; justify-content:space-between"><span>{name}</span><span>{val:.1%}</span></div>
                <div style="background-color:#334155; border-radius:10px; height:12px">
                    <div style="background-color:{color}; width:{val*100}%; height:12px; border-radius:10px; transition: width 0.5s"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # DAS WOW-FEATURE: KAUSALE ATTRIBUTION
    st.write(" ")
    with st.container():
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader("💡 KI-Erklärung (Warum?)")
        if ev:
            if ev['Vibration'] == 1 and res[2] > 0.4:
                st.write("Die KI erkennt **Vibrationen**, die untypisch für das aktuelle Alter sind. Dies deutet kausal auf einen Bruch hin.")
            elif ev['Strom'] == 1:
                st.write("Erhöhte **Stromaufnahme** detektiert. Das BN schließt daraus auf erhöhte Reibung (Stumpf).")
            else:
                st.write("Alle Sensoren im Normbereich. Die Wahrscheinlichkeit basiert auf dem **Alter**.")
        else: st.write("Warte auf Daten...")
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.subheader("📈 Wahrscheinlichkeits-Historie")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['Bruch'], name="Bruch-Gefahr", line=dict(color='#ef4444', width=3)))
        fig.add_trace(go.Scatter(x=df['t'], y=df['Stumpf'], name="Verschleiß", line=dict(color='#eab308', width=2, dash='dot')))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"), height=350,
            xaxis=dict(showgrid=False), yaxis=dict(range=[0, 1.1], gridcolor="#334155")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Noch keine Daten vorhanden. Simulation starten!")

if st.session_state.is_run:
    time.sleep(speed)
    st.rerun()
