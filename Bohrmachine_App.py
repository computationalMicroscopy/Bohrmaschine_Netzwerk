import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v9", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e0e0e0; }
    .sensor-tile { background-color: #11141a; border-radius: 8px; padding: 15px; border: 1px solid #1e293b; text-align: center; }
    .metric-value { font-family: 'IBM Plex Mono', monospace; color: #3b82f6; font-size: 1.5rem; font-weight: bold; }
    .log-container { height: 400px; overflow-y: scroll; background-color: #000; border: 1px solid #334155; padding: 10px; font-family: 'IBM Plex Mono'; font-size: 0.75rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DYNAMISCHE LOGIK-ENGINE ---
@st.cache_resource
def create_dynamic_bn(noise_v, bohrer_mat, werkst_mat):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Kuehlung', 'Zustand'), ('Vorschub_Regler', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur'), ('Zustand', 'Vorschub_Ist')
    ])
    
    # Priors (Fixiert auf Auswahl)
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_mat == "HSS" else 0.0], [1.0 if bohrer_mat == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 3, [[1.0 if werkst_mat == "Alu" else 0.0], [1.0 if werkst_mat == "Edelstahl" else 0.0], [1.0 if werkst_mat == "Titan" else 0.0]])
    cpd_k = TabularCPD('Kuehlung', 2, [[0.5], [0.5]]) # Wir lassen das Modell über den Zustand der Kühlung grübeln
    cpd_vr = TabularCPD('Vorschub_Regler', 2, [[0.5], [0.5]]) 
    
    # --- DYNAMISCHE CPT GENERIERUNG ---
    # Wir füllen 72 Spalten (3 Alter * 2 Bohrer * 3 Werkst * 2 Kühl * 2 Vorschub)
    z_matrix = []
    for a in range(3): # Alter
        for bm in range(2): # Bohrer (0=HSS, 1=HM)
            for wm in range(3): # Werkstück (0=Alu, 1=Stahl, 2=Titan)
                for k in range(2): # Kühlung (0=OK, 1=FAIL)
                    for vr in range(2): # Vorschub (0=Low, 1=High)
                        # Basis-Risiko-Score (0 bis 10)
                        score = (a * 2) + (wm * 2.5) + (k * 4) + (vr * 2) - (bm * 3)
                        score = max(0, score)
                        
                        # Wahrscheinlichkeiten basierend auf Score
                        p_bruch = min(0.99, score / 12.0)
                        p_stumpf = min(1.0 - p_bruch, (score * 0.5) / 12.0)
                        p_intakt = 1.0 - p_bruch - p_stumpf
                        z_matrix.append([p_intakt, p_stumpf, p_bruch])
    
    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Alter', 'BohrerMat', 'WerkstMat', 'Kuehlung', 'Vorschub_Regler'], 
                       evidence_card=[3, 2, 3, 2, 2])
    
    # Sensoren Likelihoods (P(Sensor|Zustand))
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.4, 0.05], [noise_v, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[0.95, 0.3, 0.1], [0.05, 0.7, 0.9]], evidence=['Zustand'], evidence_card=[3])
    cpd_vi = TabularCPD('Vorschub_Ist', 2, [[0.99, 0.4, 0.01], [0.01, 0.6, 0.99]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_bm, cpd_wm, cpd_k, cpd_vr, cpd_z, cpd_v, cpd_t, cpd_vi)
    return model

# --- 3. SESSION STATE & UI ---
if 'history' not in st.session_state:
    st.session_state.update({'count': 0, 'history': [], 'is_run': False, 'manual_fail': False, 'logs': []})

with st.sidebar:
    st.header("⚙️ Parameter")
    b_mat = st.selectbox("Bohrer", ["HSS", "Hartmetall"])
    w_mat = st.selectbox("Werkstück", ["Alu", "Edelstahl", "Titan"])
    st.divider()
    v_cut = st.slider("v_c (Schnittgeschw.)", 20, 250, 100)
    feed_in = st.slider("f (Vorschub)", 0.05, 0.8, 0.2)
    st.divider()
    instab = st.slider("Instabilität (Vib-Noise)", 0.0, 1.0, 0.1)
    k_fail = st.toggle("Kühlmittel-Ausfall")
    speed = st.select_slider("Takt", [1000, 500, 200, 50], 200)

bn = create_dynamic_bn(instab, b_mat, w_mat)
inf = VariableElimination(bn)

# --- 4. SIMULATION ---
st.title("🛡️ AI Drilling Lab v9: Deep Inference")

if st.session_state.is_run:
    st.session_state.count += 1
    # Physik-Simulation für Sensoren
    is_critical = (w_mat == "Titan" and b_mat == "HSS") or k_fail or feed_in > 0.5
    
    temp = np.random.normal(loc=(120 if k_fail else 40), scale=5) + (feed_in * 50)
    vib = np.random.normal(20, 5) + (instab * 60) + (100 if st.session_state.manual_fail else 0)
    
    # Evidenz an KI übergeben
    ev = {
        'Vibration': 1 if vib > 55 else 0,
        'Temperatur': 1 if temp > 75 else 0,
        'Vorschub_Ist': 1 if st.session_state.manual_fail else 0,
        'BohrerMat': 0 if b_mat == "HSS" else 1,
        'WerkstMat': ["Alu", "Edelstahl", "Titan"].index(w_mat),
        'Kuehlung': 1 if k_fail else 0,
        'Vorschub_Regler': 1 if feed_in > 0.3 else 0,
        'Alter': min(2, st.session_state.count // 30)
    }
    res = inf.query(['Zustand'], evidence=ev).values
    
    # Reset manual fail nach Berechnung
    st.session_state.manual_fail = False
    
    st.session_state.history.append({'t': st.session_state.count, 'prob': res[2], 'temp': temp, 'vib': vib})
    st.session_state.logs.insert(0, f"Cycle {st.session_state.count}: P(Bruch)={res[2]:.1%} | T={temp:.1f}°C")
else:
    res = [1, 0, 0]; temp = 0; vib = 0

# --- 5. DASHBOARD ---
c1, c2, c3 = st.columns([1, 1, 1])
with c1: st.button("▶️ START/STOP", on_click=lambda: setattr(st.session_state, 'is_run', not st.session_state.is_run), use_container_width=True)
with c2: st.button("🧹 RESET", on_click=lambda: st.session_state.update({'count':0,'history':[],'logs':[]}), use_container_width=True)
with c3: st.button("💥 FEHLER", on_click=lambda: setattr(st.session_state, 'manual_fail', True), use_container_width=True)

st.write("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['prob'], name="Bruchrisiko", fill='tozeroy', line_color='red'))
        fig.update_layout(height=400, template="plotly_dark", title="KI-Inferenz: Wahrscheinlichkeit für Werkzeugbruch")
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📜 Live-Analyse")
    log_h = "".join([f"<div>{l}</div>" for l in st.session_state.logs[:50]])
    st.markdown(f'<div class="log-container">{log_h}</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(speed/1000)
    st.rerun()
