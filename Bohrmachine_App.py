import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v12", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e0e0e0; }
    .sensor-tile { background-color: #11141a; border-radius: 8px; padding: 15px; border: 1px solid #1e293b; text-align: center; }
    .metric-value { font-family: 'IBM Plex Mono', monospace; color: #3b82f6; font-size: 1.5rem; font-weight: bold; }
    .log-container { height: 450px; overflow-y: scroll; background-color: #000; border: 1px solid #334155; padding: 10px; font-family: 'IBM Plex Mono'; font-size: 0.75rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DYNAMISCHE LOGIK (Mathematik korrigiert) ---
@st.cache_resource
def create_life_cycle_bn(n_v, n_t, n_s, bohrer_mat, werkst_mat):
    model = DiscreteBayesianNetwork([
        ('Verschleiss', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Kuehlung', 'Zustand'), ('Vorschub_Regler', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur'), ('Zustand', 'Strom'),
        ('Zustand', 'Drehmoment'), ('Zustand', 'Vorschub_Ist')
    ])
    
    # Priors
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_mat == "HSS" else 0.0], [1.0 if bohrer_mat == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 3, [[1.0 if werkst_mat == "Alu" else 0.0], [1.0 if werkst_mat == "Edelstahl" else 0.0], [1.0 if werkst_mat == "Titan" else 0.0]])
    cpd_k = TabularCPD('Kuehlung', 2, [[0.5], [0.5]]) 
    cpd_vr = TabularCPD('Vorschub_Regler', 2, [[0.5], [0.5]]) 
    cpd_vschleiss = TabularCPD('Verschleiss', 3, [[0.33], [0.33], [0.34]])

    # Dynamische CPT für Zustand (72 Spalten)
    z_matrix = []
    for v in range(3):
        for bm in range(2): 
            for wm in range(3): 
                for k in range(2): 
                    for vr in range(2):
                        score = (v * 4.0) + (wm * 3.5) + (k * 6.0) + (vr * 3.0) - (bm * 4.0)
                        score = max(0.2, score)
                        p_bruch = min(0.99, (score**2) / 250.0)
                        p_stumpf = min(1.0 - p_bruch, (score * 0.8) / 20.0)
                        p_intakt = 1.0 - p_bruch - p_stumpf
                        z_matrix.append([p_intakt, p_stumpf, p_bruch])
    
    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Verschleiss', 'BohrerMat', 'WerkstMat', 'Kuehlung', 'Vorschub_Regler'], 
                       evidence_card=[3, 2, 3, 2, 2])
    
    # Sensoren Likelihoods (Korrektur: Jede Liste braucht 3 Werte für Zustand [0,1,2])
    cpd_vib = TabularCPD('Vibration', 2, [[1-n_v, 0.4, 0.05], [n_v, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_temp = TabularCPD('Temperatur', 2, [[1-n_t, 0.3, 0.1], [n_t, 0.7, 0.9]], evidence=['Zustand'], evidence_card=[3])
    cpd_strom = TabularCPD('Strom', 2, [[1-n_s, 0.2, 0.4], [n_s, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_dreh = TabularCPD('Drehmoment', 2, [[0.9, 0.2, 0.5], [0.1, 0.8, 0.5]], evidence=['Zustand'], evidence_card=[3])
    cpd_vi = TabularCPD('Vorschub_Ist', 2, [[0.99, 0.4, 0.01], [0.01, 0.6, 0.99]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_bm, cpd_wm, cpd_k, cpd_vr, cpd_vschleiss, cpd_z, cpd_vib, cpd_temp, cpd_strom, cpd_dreh, cpd_vi)
    return model

# --- 3. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.update({
        'count': 0, 'history': [], 'is_run': False, 'manual_fail': False, 
        'logs': [], 'cum_wear': 0.0, 'has_broken': False
    })

# --- 4. SIDEBAR (Alle Regler wiederhergestellt) ---
with st.sidebar:
    st.title("🔩 CNC Control Unit")
    b_mat = st.selectbox("Bohrer", ["HSS", "Hartmetall"])
    w_mat = st.selectbox("Werkstück", ["Alu", "Edelstahl", "Titan"])
    st.divider()
    v_cut = st.slider("Schnittgeschw. v_c (m/min)", 20, 250, 100)
    f_in = st.slider("Vorschub f (mm/U)", 0.05, 0.8, 0.2)
    st.divider()
    with st.expander("Sensor-Rauschen & Stabilität"):
        n_v = st.slider("Vibrations-Noise", 0.0, 1.0, 0.1)
        n_t = st.slider("Temperatur-Noise", 0.0, 1.0, 0.05)
        n_s = st.slider("Strom-Rauschen", 0.0, 1.0, 0.02)
        instability = st.slider("Instabile Aufspannung", 0.0, 1.0, 0.1)
    k_fail = st.toggle("Kühlmittel-Ausfall")
    st_speed = st.select_slider("Sim-Takt", [1000, 500, 200, 50, 10], 200)

bn = create_life_cycle_bn(n_v, n_t, n_s, b_mat, w_mat)
inf = VariableElimination(bn)

# --- 5. PHYSIK & STANDZEIT-BERECHNUNG ---
if st.session_state.is_run and not st.session_state.has_broken:
    st.session_state.count += 1
    
    # Stressfaktoren (Wichtig für reale Standzeit)
    material_factor = {"Alu": 0.1, "Edelstahl": 1.0, "Titan": 5.0}[w_mat]
    bohrer_factor = 0.5 if b_mat == "Hartmetall" else 2.5
    coolant_stress = 10.0 if k_fail else 1.0
    feed_stress = (f_in / 0.2)**2
    
    # Kumulativer Verschleiß
    wear_inc = 0.01 * material_factor * bohrer_factor * coolant_stress * feed_stress
    st.session_state.cum_wear += wear_inc
    
    # Verschleiß-Knoten für KI
    wear_idx = 0 if st.session_state.cum_wear < 30 else (1 if st.session_state.cum_wear < 85 else 2)
    
    # Sensoren
    vib = np.random.normal(20, 5) + (instability * 50) + (st.session_state.cum_wear * 0.4)
    temp = np.random.normal(40, 4) + (st.session_state.cum_wear * 0.7) + (100 if k_fail else 0) + (v_cut * 0.1)
    f_ist = f_in if not st.session_state.manual_fail else (f_in * 0.05)
    
    # KI-Inferenz
    ev = {
        'Verschleiss': wear_idx, 'Kuehlung': 1 if k_fail else 0,
        'BohrerMat': 0 if b_mat == "HSS" else 1, 'WerkstMat': ["Alu", "Edelstahl", "Titan"].index(w_mat),
        'Vorschub_Regler': 1 if f_in > 0.3 else 0, 'Vibration': 1 if vib > 60 else 0,
        'Temperatur': 1 if temp > 80 else 0, 'Vorschub_Ist': 1 if st.session_state.manual_fail else 0
    }
    res = inf.query(['Zustand'], evidence=ev).values
    
    if res[2] > 0.98 or st.session_state.cum_wear > 150 or st.session_state.manual_fail:
        st.session_state.has_broken = True
        st.session_state.is_run = False
        st.session_state.manual_fail = False

    st.session_state.history.append({'t': st.session_state.count, 'prob': res[2], 'wear': st.session_state.cum_wear, 'vib': vib, 'temp': temp})
    st.session_state.logs.insert(0, f"Cycle {st.session_state.count}: Verschleiß {st.session_state.cum_wear:.1f}% | Risk {res[2]:.1%}")
else:
    res = [1.0, 0.0, 0.0]

# --- 6. DISPLAY ---
st.title("🛡️ AI Industrial Twin: Standzeit-Labor")
if st.session_state.has_broken:
    st.error(f"💥 WERKZEUGBRUCH! Standzeit nach {st.session_state.count} Zyklen erreicht.")

c1, c2, c3 = st.columns(3)
with c1: 
    if st.button("▶️ START/STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with c2:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.update({'count':0,'history':[],'is_run':False,'logs':[],'cum_wear':0.0,'has_broken':False})
        st.rerun()
with c3:
    st.metric("Verschleiß-Fortschritt", f"{st.session_state.cum_wear:.1f}%")

st.write("---")
col_l, col_r = st.columns([2, 1])

with col_l:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['wear'], name="Real-Verschleiß (%)", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df['t'], y=df['prob']*100, name="KI-Bruchrisiko (%)", line=dict(color='red', width=3)))
        fig.update_layout(height=400, template="plotly_dark", title="Physikalischer Verschleiß vs. KI-Inferenz")
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("📜 Event-Log")
    log_html = "".join([f"<div>{l}</div>" for l in st.session_state.logs[:100]])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(st_speed/1000)
    st.rerun()
