import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v7", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e0e0e0; }
    .sensor-tile { 
        background-color: #11141a; border-radius: 8px; padding: 15px; 
        border: 1px solid #1e293b; text-align: center;
    }
    .metric-value { font-family: 'IBM Plex Mono', monospace; color: #3b82f6; font-size: 1.5rem; font-weight: bold; }
    .log-container { 
        height: 450px; overflow-y: scroll; background-color: #000000; 
        border: 1px solid #334155; padding: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DAS EXPERTEN-MODELL (Vollständige Kausalität) ---
@st.cache_resource
def create_full_expert_bn(noise_v, noise_t, noise_s, bohrer_mat, werkst_mat):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Kuehlung', 'Zustand'), ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Drehmoment'), ('Zustand', 'Akustik')
    ])
    
    # Priors
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_mat == "HSS" else 0.0], [1.0 if bohrer_mat == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 3, [[1.0 if werkst_mat == "Alu" else 0.0], [1.0 if werkst_mat == "Edelstahl" else 0.0], [1.0 if werkst_mat == "Titan" else 0.0]])
    cpd_k = TabularCPD('Kuehlung', 2, [[0.95], [0.05]]) 
    
    # Zustand CPT (Basis-Matrix)
    v_z = np.zeros((3, 36))
    v_z[0, :] = 0.8; v_z[1, :] = 0.15; v_z[2, :] = 0.05
    cpd_z = TabularCPD('Zustand', 3, v_z, evidence=['Alter', 'BohrerMat', 'WerkstMat', 'Kuehlung'], evidence_card=[3, 2, 3, 2])
    
    # Sensoren mit individuellen Noise-Parametern
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.3, 0.05], [noise_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[1-noise_t, 0.2, 0.1], [noise_t, 0.8, 0.9]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[1-noise_s, 0.2, 0.4], [noise_s, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    
    # Standard-Sensoren
    cpd_d = TabularCPD('Drehmoment', 2, [[0.9, 0.2, 0.5], [0.1, 0.8, 0.5]], evidence=['Zustand'], evidence_card=[3])
    cpd_ak = TabularCPD('Akustik', 2, [[0.9, 0.3, 0.4], [0.1, 0.7, 0.6]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_bm, cpd_wm, cpd_k, cpd_z, cpd_v, cpd_s, cpd_t, cpd_d, cpd_ak)
    return model

# --- 3. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.update({'count': 0, 'history': [], 'is_run': False, 'manual_fail': False, 'logs': []})

# --- 4. SIDEBAR (Alle Regler) ---
with st.sidebar:
    st.title("🛠️ CNC-Steuerpult")
    
    with st.expander("Werkzeug & Werkstück", expanded=True):
        b_mat = st.selectbox("Bohrer-Material", ["HSS", "Hartmetall"])
        w_mat = st.selectbox("Werkstück-Material", ["Alu", "Edelstahl", "Titan"])
        v_cut = st.slider("Schnittgeschw. v_c (m/min)", 20, 250, 100)
    
    with st.expander("Sensor-Kalibrierung (Rauschen)", expanded=False):
        n_v = st.slider("Vibrations-Noise", 0.0, 1.0, 0.1)
        n_t = st.slider("Temperatur-Noise", 0.0, 1.0, 0.05)
        n_s = st.slider("Strom-Rauschen", 0.0, 1.0, 0.02)
    
    with st.expander("Prozess-Störungen", expanded=True):
        instability = st.slider("Instabile Aufspannung", 0.0, 1.0, 0.1)
        k_fail = st.toggle("Kühlmittel-Ausfall")
        st_speed = st.select_slider("Sim-Frequenz", [1000, 500, 200, 50, 10], 200)

bn = create_full_expert_bn(n_v, n_t, n_s, b_mat, w_mat)
inf = VariableElimination(bn)

# --- 5. MAIN DASHBOARD ---
st.title("🛡️ AI-Industrial Twin: High-Fidelity Professional")

c_b1, c_b2, c_b3, c_b4 = st.columns([1, 1, 1, 2])
with c_b1:
    if st.button("▶️ START / STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with c_b2:
    if st.button("🔄 LABOR-RESET", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'logs':[]})
        st.rerun()
with c_b3:
    if st.button("💥 BRUCH ERZWINGEN", type="primary", use_container_width=True): st.session_state.manual_fail = True
with c_b4:
    st.subheader(f"Zyklen: {st.session_state.count}")

# --- 6. PHYSIK-SIMULATION ---
if st.session_state.is_run:
    st.session_state.count += 1
    age_idx = min(2, st.session_state.count // 40)
    
    # Bruch-Wahrscheinlichkeit basierend auf Physik
    if st.session_state.manual_fail:
        true_z = 2
        st.session_state.manual_fail = False
    else:
        base_risk = {"Alu": 0.001, "Edelstahl": 0.01, "Titan": 0.04}[w_mat]
        risk_mult = 5.0 if k_fail else 1.0
        true_z = 2 if np.random.random() < (base_risk * risk_mult * (age_idx + 1)) else (1 if np.random.random() < 0.15 else 0)

    # Sensor-Signale generieren
    vib = np.random.normal(20, 5) + (instability * 80 * np.random.random()) + (80 if true_z==2 else 0)
    temp = np.random.normal(loc=(100 if k_fail else 40), scale=4) + (true_z * 25) + (v_cut * 0.1)
    torque = (v_cut * 0.3) + (50 if true_z > 0 else 0) + np.random.normal(0, 2)
    
    # KI-Inferenz
    ev = {
        'Vibration': 1 if vib > 60 else 0,
        'Temperatur': 1 if temp > 75 else 0,
        'Kuehlung': 1 if k_fail else 0,
        'BohrerMat': 0 if b_mat == "HSS" else 1,
        'WerkstMat': ["Alu", "Edelstahl", "Titan"].index(w_mat),
        'Alter': age_idx
    }
    res = inf.query(['Zustand'], evidence=ev).values
    
    # Logging
    ts = time.strftime("%H:%M:%S")
    log_entry = f"[{ts}] V:{vib:.1f}g | T:{temp:.1f}°C | Torque:{torque:.1f}Nm | Risk:{res[2]:.1%}"
    st.session_state.logs.insert(0, log_entry)
    st.session_state.history.append({'t': st.session_state.count, 'prob': res[2], 'vib': vib, 'temp': temp, 'torque': torque})
else:
    res, vib, temp, torque = [1, 0, 0], 0, 0, 0

# --- 7. DASHBOARD LAYOUT ---
st.write("---")
col_tel, col_plot, col_log = st.columns([1, 2, 1.2])

with col_tel:
    st.subheader("📡 Telemetrie")
    st.markdown(f'<div class="sensor-tile">VIB (g)<br><span class="metric-value">{vib:.1f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile" style="margin-top:10px;">TEMP (°C)<br><span class="metric-value" style="color:#f97316;">{temp:.1f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile" style="margin-top:10px;">DREHMOMENT (Nm)<br><span class="metric-value" style="color:#10b981;">{torque:.1f}</span></div>', unsafe_allow_html=True)
    
    st.write("---")
    st.subheader("🧠 KI-Diagnose")
    for s, p, c in zip(["Intakt", "Verschleiß", "Bruch"], res, ["#10b981", "#f59e0b", "#ef4444"]):
        st.caption(f"{s}: {p:.1%}")
        st.progress(float(p))

with col_plot:
    st.subheader("📊 Prozess-Historie")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['vib'], name="Vibration (g)", line=dict(color='#3b82f6')))
        fig.add_trace(go.Scatter(x=df['t'], y=df['prob']*100, name="Bruch-Risiko (%)", line=dict(color='#ef4444', width=3), fill='tozeroy'))
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_log:
    st.subheader("📜 XAI Terminal")
    log_html = "".join([f'<div style="border-bottom: 1px solid #1e293b; padding: 2px;">{l}</div>' for l in st.session_state.logs[:100]])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(st_speed/1000)
    st.rerun()
