import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. PRO-LEVEL UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v10", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e0e0e0; }
    .sensor-tile { 
        background-color: #11141a; border-radius: 8px; padding: 15px; 
        border: 1px solid #1e293b; text-align: center;
    }
    .metric-value { font-family: 'IBM Plex Mono', monospace; color: #3b82f6; font-size: 1.5rem; font-weight: bold; }
    .log-container { 
        height: 500px; overflow-y: scroll; background-color: #000000; 
        border: 1px solid #334155; padding: 10px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DIE KOMPLEXE DYNAMISCHE LOGIK (Mathematisch plausibel) ---
@st.cache_resource
def create_full_expert_bn(n_v, n_t, n_s, bohrer_mat, werkst_mat):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Kuehlung', 'Zustand'), ('Vorschub_Regler', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Drehmoment'), ('Zustand', 'Vorschub_Ist')
    ])
    
    # Priors
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_mat == "HSS" else 0.0], [1.0 if bohrer_mat == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 3, [[1.0 if werkst_mat == "Alu" else 0.0], [1.0 if werkst_mat == "Edelstahl" else 0.0], [1.0 if werkst_mat == "Titan" else 0.0]])
    cpd_k = TabularCPD('Kuehlung', 2, [[0.5], [0.5]]) 
    cpd_vr = TabularCPD('Vorschub_Regler', 2, [[0.5], [0.5]]) 

    # Dynamische CPT Berechnung (72 Kombinationen)
    z_matrix = []
    for a in range(3): # Alter
        for bm in range(2): # Bohrer (0=HSS, 1=HM)
            for wm in range(3): # Werkst (0=Alu, 1=Stahl, 2=Titan)
                for k in range(2): # Kühl (0=OK, 1=FAIL)
                    for vr in range(2): # Vorschub (0=Low, 1=High)
                        # Score-basiert für Profi-Plausibilität
                        score = (a * 2.0) + (wm * 3.0) + (k * 5.0) + (vr * 3.0) - (bm * 3.5)
                        score = max(0.5, score)
                        p_bruch = min(0.98, score / 15.0)
                        p_stumpf = min(1.0 - p_bruch, (score * 0.6) / 15.0)
                        p_intakt = 1.0 - p_bruch - p_stumpf
                        z_matrix.append([p_intakt, p_stumpf, p_bruch])
    
    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Alter', 'BohrerMat', 'WerkstMat', 'Kuehlung', 'Vorschub_Regler'], 
                       evidence_card=[3, 2, 3, 2, 2])
    
    # Sensoren mit individuellem Noise
    cpd_v = TabularCPD('Vibration', 2, [[1-n_v, 0.3, 0.05], [n_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[1-n_t, 0.2, 0.1], [n_t, 0.8, 0.9]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[1-n_s, 0.2, 0.4], [n_s, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_d = TabularCPD('Drehmoment', 2, [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]], evidence=['Zustand'], evidence_card=[3])
    cpd_vi = TabularCPD('Vorschub_Ist', 2, [[0.99, 0.4, 0.01], [0.01, 0.6, 0.99]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_bm, cpd_wm, cpd_k, cpd_vr, cpd_z, cpd_v, cpd_s, cpd_t, cpd_d, cpd_vi)
    return model

# --- 3. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.update({'count': 0, 'history': [], 'is_run': False, 'manual_fail': False, 'logs': []})

# --- 4. SIDEBAR (Alle Profi-Regler wiederhergestellt) ---
with st.sidebar:
    st.title("🔩 CNC-Steuerung")
    
    with st.expander("Werkzeug & Material", expanded=True):
        b_mat = st.selectbox("Bohrer-Material", ["HSS", "Hartmetall"])
        w_mat = st.selectbox("Werkstück-Material", ["Alu", "Edelstahl", "Titan"])
        v_cut = st.slider("v_c (Schnittgeschw. m/min)", 20, 250, 100)
        f_in = st.slider("f (Vorschub mm/U)", 0.05, 0.8, 0.2)
    
    with st.expander("Sensor-Kalibrierung (Rauschen)", expanded=False):
        noise_v = st.slider("Vibrations-Noise", 0.0, 1.0, 0.1)
        noise_t = st.slider("Temperatur-Noise", 0.0, 1.0, 0.05)
        noise_s = st.slider("Strom-Rauschen", 0.0, 1.0, 0.02)
    
    with st.expander("Prozess-Stabilität", expanded=True):
        instability = st.slider("Instabile Aufspannung", 0.0, 1.0, 0.1)
        k_fail = st.toggle("Kühlmittel-Ausfall")
        st_speed = st.select_slider("Simulations-Takt", [1000, 500, 200, 50, 10], 200)

bn = create_full_expert_bn(noise_v, noise_t, noise_s, b_mat, w_mat)
inf = VariableElimination(bn)

# --- 5. MAIN DASHBOARD ---
st.title("🛡️ AI Industrial Twin: Professional Simulation Lab")

c_b1, c_b2, c_b3, c_b4 = st.columns([1, 1, 1, 2])
with c_b1:
    if st.button("▶️ START / STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with c_b2:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'logs':[]})
        st.rerun()
with c_b3:
    if st.button("💥 FEHLER ERZWINGEN", type="primary", use_container_width=True): st.session_state.manual_fail = True
with c_b4:
    st.subheader(f"Prozesszyklen: {st.session_state.count}")

# --- 6. PHYSIK & KI-INFERENZ ---
if st.session_state.is_run:
    st.session_state.count += 1
    age_idx = min(2, st.session_state.count // 40)
    
    # Sensordaten generieren (Physik-Simulation)
    # Vibration: Basis + Instabilität + Schaden
    vib = np.random.normal(20, 5) + (instability * 70 * np.random.random()) + (90 if st.session_state.manual_fail else 0)
    # Temperatur: Basis + Kühlung + Vorschub + v_c
    temp = np.random.normal(loc=(110 if k_fail else 40), scale=4) + (f_in * 40) + (v_cut * 0.1)
    # Drehmoment
    torque = (v_cut * f_in * 1.5) + (50 if st.session_state.manual_fail else 0)
    # Vorschub-Ist (sinkt bei Fehler/Bruch)
    f_ist = f_in if not st.session_state.manual_fail else (f_in * 0.1)
    
    # KI-Evidenz
    ev = {
        'Vibration': 1 if vib > 60 else 0,
        'Temperatur': 1 if temp > 75 else 0,
        'Kuehlung': 1 if k_fail else 0,
        'Vorschub_Regler': 1 if f_in > 0.3 else 0,
        'Vorschub_Ist': 1 if f_ist < (f_in * 0.5) else 0,
        'BohrerMat': 0 if b_mat == "HSS" else 1,
        'WerkstMat': ["Alu", "Edelstahl", "Titan"].index(w_mat),
        'Alter': age_idx
    }
    res = inf.query(['Zustand'], evidence=ev).values
    
    # Reset manual fail nach Inferenz
    st.session_state.manual_fail = False
    
    # Logging & History
    ts = time.strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{ts}] Cycle {st.session_state.count}: P(Bruch)={res[2]:.1%} | f={f_ist:.2f} | T={temp:.1f}°C")
    st.session_state.history.append({'t': st.session_state.count, 'prob': res[2], 'vib': vib, 'temp': temp, 'torque': torque, 'f': f_ist})
else:
    res, vib, temp, torque, f_ist = [1, 0, 0], 0, 0, 0, 0

# --- 7. DISPLAY LAYOUT ---
st.write("---")
col_tel, col_plot, col_log = st.columns([1.2, 2, 1.2])

with col_tel:
    st.subheader("📡 Telemetrie")
    st.markdown(f'<div class="sensor-tile">VORSCHUB f (mm/U)<br><span class="metric-value" style="color:#a855f7;">{f_ist:.2f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile" style="margin-top:10px;">TEMPERATUR (°C)<br><span class="metric-value" style="color:#f97316;">{temp:.1f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile" style="margin-top:10px;">VIBRATION (g)<br><span class="metric-value">{vib:.1f}</span></div>', unsafe_allow_html=True)
    
    st.write("---")
    st.subheader("🧠 KI-Status")
    for label, prob, color in zip(["Intakt", "Stumpf", "Bruch"], res, ["#10b981", "#f59e0b", "#ef4444"]):
        st.caption(f"{label}: {prob:.1%}")
        st.progress(float(prob))

with col_plot:
    st.subheader("📊 Inferenz-Historie")
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
