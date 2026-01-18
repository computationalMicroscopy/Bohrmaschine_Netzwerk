import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. PRO-LEVEL UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v6", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e0e0e0; }
    .sensor-tile { 
        background-color: #11141a; border-radius: 8px; padding: 15px; 
        border: 1px solid #1e293b; text-align: center;
    }
    .log-container { 
        height: 400px; overflow-y: scroll; background-color: #000000; 
        border: 1px solid #334155; padding: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DAS KOMPLEXE BN-MODELL (Vibration als steuerbarer Einfluss) ---
@st.cache_resource
def create_expert_bn(noise_v, bohrer_mat, werkst_mat):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Kuehlung', 'Zustand'), ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Drehmoment'), ('Zustand', 'Akustik')
    ])
    
    # Priors & Definitionen
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_mat == "HSS" else 0.0], [1.0 if bohrer_mat == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 3, [[1.0 if werkst_mat == "Alu" else 0.0], [1.0 if werkst_mat == "Edelstahl" else 0.0], [1.0 if werkst_mat == "Titan" else 0.0]])
    cpd_k = TabularCPD('Kuehlung', 2, [[0.95], [0.05]]) 
    
    # Zustand CPT (Basis-Setup)
    v_z = np.zeros((3, 36))
    v_z[0, :] = 0.8; v_z[1, :] = 0.15; v_z[2, :] = 0.05 # Standard-Verteilung
    cpd_z = TabularCPD('Zustand', 3, v_z, evidence=['Alter', 'BohrerMat', 'WerkstMat', 'Kuehlung'], evidence_card=[3, 2, 3, 2])
    
    # Der Vibrations-Knoten reagiert nun sensibler auf das eingestellte Rauschen
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.3, 0.05], [noise_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[0.9, 0.2, 0.4], [0.1, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[0.95, 0.2, 0.4], [0.05, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_d = TabularCPD('Drehmoment', 2, [[0.9, 0.2, 0.5], [0.1, 0.8, 0.5]], evidence=['Zustand'], evidence_card=[3])
    cpd_ak = TabularCPD('Akustik', 2, [[0.9, 0.3, 0.4], [0.1, 0.7, 0.6]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_bm, cpd_wm, cpd_k, cpd_z, cpd_v, cpd_s, cpd_t, cpd_d, cpd_ak)
    return model

# --- 3. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.update({
        'count': 0, 'history': [], 'is_run': False, 'manual_fail': False,
        'logs': []
    })

# --- 4. SIDEBAR (Profi-Regler) ---
with st.sidebar:
    st.header("⚙️ Maschinen-Setup")
    b_mat = st.selectbox("Bohrer", ["HSS", "Hartmetall"])
    w_mat = st.selectbox("Werkstück", ["Alu", "Edelstahl", "Titan"])
    
    st.divider()
    st.header("⚡ Prozess-Stabilität")
    # Der neue Vibrations-Einfluss-Regler
    vib_env = st.slider("Mechanische Instabilität", 0.0, 1.0, 0.1, help="Simuliert Resonanzen oder schlechte Aufspannung.")
    k_fail = st.toggle("Kühlmittel-System AUS")
    
    st.divider()
    speed = st.select_slider("Taktung", [1000, 500, 200, 50, 10], 200)

bn = create_expert_bn(vib_env, b_mat, w_mat)
inf = VariableElimination(bn)

# --- 5. MAIN DASHBOARD ---
st.title("🔩 AI Industrial Lab: High-Fidelity Simulation")

c_btn1, c_btn2, c_btn3, c_info = st.columns([1, 1, 1, 2])
with c_btn1:
    if st.button("🚀 START/STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with c_btn2:
    if st.button("🧹 RESET", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'logs':[]})
        st.rerun()
with c_btn3:
    if st.button("⚠️ BRUCH ERZWINGEN", type="primary", use_container_width=True): st.session_state.manual_fail = True

# --- 6. LOGIK & PHYSIK ---
if st.session_state.is_run:
    st.session_state.count += 1
    age_idx = min(2, st.session_state.count // 40)
    
    # Schadens-Logik
    if st.session_state.manual_fail:
        true_z = 2
        st.session_state.manual_fail = False
    else:
        # Basis-Risiko steigt mit Alter und Materialhärte
        risk_map = {"Alu": 0.005, "Edelstahl": 0.02, "Titan": 0.05}
        true_z = 2 if np.random.random() < risk_map[w_mat] * (age_idx + 1) else (1 if np.random.random() < 0.1 else 0)

    # Physikalische Signal-Generierung
    # Grund-Vibration + Instabilitäts-Einfluss + Schaden
    base_vib = np.random.normal(20, 5)
    env_vib = vib_env * 60 * np.random.random() # Fluktuation durch Instabilität
    damage_vib = 80 if true_z == 2 else (30 if true_z == 1 else 0)
    final_vib = base_vib + env_vib + damage_vib
    
    temp = np.random.normal(loc=(90 if k_fail else 35), scale=3) + (true_z * 20)
    
    # KI Evidenz
    ev = {
        'Vibration': 1 if final_vib > 50 else 0,
        'Temperatur': 1 if temp > 60 else 0,
        'Kuehlung': 1 if k_fail else 0,
        'BohrerMat': 0 if b_mat == "HSS" else 1,
        'WerkstMat': ["Alu", "Edelstahl", "Titan"].index(w_mat),
        'Alter': age_idx
    }
    res = inf.query(['Zustand'], evidence=ev).values

    # History & Logs
    st.session_state.logs.insert(0, f"Cycle {st.session_state.count:03d}: V={final_vib:.1f}g | T={temp:.1f}°C | P(Bruch)={res[2]:.1%}")
    st.session_state.history.append({'t': st.session_state.count, 'prob': res[2], 'vib': final_vib, 'temp': temp})
else:
    res, final_vib, temp = [1, 0, 0], 0, 0

# --- 7. VISUALISIERUNG ---
st.write("---")
col_metrics, col_graph, col_log = st.columns([1, 2, 1.2])

with col_metrics:
    st.subheader("📡 Live-Telemetrie")
    st.markdown(f'<div class="sensor-tile"><small>VIBRATION (g)</small><br><span class="metric-value">{final_vib:.1f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile" style="margin-top:10px;"><small>TEMPERATUR (°C)</small><br><span class="metric-value">{temp:.1f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile" style="margin-top:10px;"><small>BRUCH-RISIKO</small><br><span class="metric-value" style="color:#ef4444;">{res[2]:.1%}</span></div>', unsafe_allow_html=True)
    
    with st.expander("Kausale Faktoren"):
        st.write(f"Vibrations-Offset: +{vib_env*100:.0f}%")
        st.write(f"Kühlung: {'AUS' if k_fail else 'OK'}")

with col_graph:
    st.subheader("📊 Prozess-Analyse")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['vib'], name="Vibration (g)", line=dict(color='#3b82f6')))
        fig.add_trace(go.Scatter(x=df['t'], y=df['prob']*100, name="Bruch-Risiko (%)", line=dict(color='#ef4444', width=3), fill='tozeroy'))
        fig.update_layout(height=380, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

with col_log:
    st.subheader("📜 XAI Terminal")
    log_html = "".join([f'<div style="border-bottom: 1px solid #1e293b; padding: 2px;">{l}</div>' for l in st.session_state.logs[:50]])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(speed/1000)
    st.rerun()
