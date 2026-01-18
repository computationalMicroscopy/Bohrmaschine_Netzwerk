import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. UI & THEME SETUP ---
st.set_page_config(layout="wide", page_title="Industrial AI Lab v5", page_icon="🏗️")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .sensor-tile { 
        background-color: #161b22; border-radius: 10px; padding: 12px; 
        border: 1px solid #30363d; margin-bottom: 8px;
    }
    .log-container {
        height: 350px; overflow-y: scroll; background-color: #0d1117;
        border: 1px solid #30363d; border-radius: 10px; padding: 15px;
        font-family: 'Courier New', Courier, monospace; font-size: 0.85rem;
    }
    .log-entry { margin-bottom: 5px; border-bottom: 1px solid #21262d; padding-bottom: 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ERWEITERTES BAYESSCHES NETZWERK (Material-Logik) ---
@st.cache_resource
def create_material_bn(noise_v, noise_c, noise_t, bohrer_typ, werkst_typ):
    # Struktur: Alter, Bohrer-Material, Werkstück -> Zustand -> Sensoren
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Akustik'),
        ('Zustand', 'Drehmoment'), ('Zustand', 'Vorschub')
    ])
    
    # Priors (Werden durch die UI-Wahl festgesetzt)
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_typ == "HSS" else 0.0], [1.0 if bohrer_typ == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 2, [[1.0 if werkst_typ == "Aluminium" else 0.0], [1.0 if werkst_typ == "Edelstahl" else 0.0]])
    
    # Zustand CPT (Komplexe Matrix: 3x2x2 = 12 Kombinationen)
    # Die Werte repräsentieren P(Zustand | Alter, BohrerMat, WerkstMat)
    # Wir vereinfachen hier die Logik: HSS in Edelstahl = Hohes Risiko / Hartmetall in Alu = Sicher
    cpd_z = TabularCPD('Zustand', 3, [
            # Intakt: Höher bei Hartmetall & Alu
            [0.99, 0.95, 0.80, 0.50, 0.90, 0.70, 0.40, 0.10, 0.70, 0.40, 0.10, 0.01], 
            # Stumpf: Steigt schnell bei HSS & Edelstahl
            [0.01, 0.04, 0.15, 0.30, 0.09, 0.20, 0.40, 0.50, 0.25, 0.40, 0.50, 0.40], 
            # Bruch: Höchstes Risiko bei altem HSS in Edelstahl
            [0.00, 0.01, 0.05, 0.20, 0.01, 0.10, 0.20, 0.40, 0.05, 0.20, 0.40, 0.59]], 
            evidence=['Alter', 'BohrerMat', 'WerkstMat'], evidence_card=[3, 2, 2])
    
    # Sensoren (Bleiben stabil in der Logik)
    cpd_v = TabularCPD('Vibration', 2, [[1-noise_v, 0.3, 0.05], [noise_v, 0.7, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_s = TabularCPD('Strom', 2, [[0.95, 0.2, 0.4], [0.05, 0.8, 0.6]], evidence=['Zustand'], evidence_card=[3])
    cpd_t = TabularCPD('Temperatur', 2, [[0.99, 0.1, 0.3], [0.01, 0.9, 0.7]], evidence=['Zustand'], evidence_card=[3])
    cpd_ak = TabularCPD('Akustik', 2, [[0.90, 0.4, 0.5], [0.10, 0.6, 0.5]], evidence=['Zustand'], evidence_card=[3])
    cpd_dr = TabularCPD('Drehmoment', 2, [[0.95, 0.3, 0.6], [0.05, 0.7, 0.4]], evidence=['Zustand'], evidence_card=[3])
    cpd_vo = TabularCPD('Vorschub', 2, [[0.98, 0.4, 0.1], [0.02, 0.6, 0.9]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_a, cpd_bm, cpd_wm, cpd_z, cpd_v, cpd_s, cpd_t, cpd_ak, cpd_dr, cpd_vo)
    return model

# --- 3. INITIALISIERUNG ---
if 'history' not in st.session_state:
    st.session_state.update({
        'count': 0, 'history': [], 'is_run': False, 
        'manual_fail': False, 'emergency_stop': False, 'event_logs': []
    })

with st.sidebar:
    st.header("🛠️ Werkzeug-Setup")
    bohrer_sel = st.selectbox("Bohrermaterial", ["HSS", "Hartmetall"], help="HSS ist günstiger, verschleißt aber schneller.")
    werkst_sel = st.selectbox("Werkstückmaterial", ["Aluminium", "Edelstahl"], help="Edelstahl erfordert höhere Kräfte und erzeugt mehr Hitze.")
    
    st.divider()
    st.header("⚙️ Sensor-Feinjustierung")
    n_v = st.slider("Vibrations-Rauschen", 0.0, 0.4, 0.05)
    n_t = st.slider("Temperatur-Rauschen", 0.0, 0.4, 0.01)
    
    st.divider()
    speed = st.select_slider("Simulations-Takt", [1.0, 0.5, 0.2, 0.1, 0.01], 0.1)
    auto_stop_active = st.checkbox("Automatischer Not-Halt", value=True)

bn = create_material_bn(n_v, 0.02, n_t, bohrer_sel, werkst_sel)
inf = VariableElimination(bn)

# --- 4. HEADER ---
st.title("🛡️ AI Industrial Twin - Expert Edition")
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
with c1:
    if st.button("▶️ START / STOP", use_container_width=True): 
        st.session_state.is_run = not st.session_state.is_run
        st.session_state.emergency_stop = False
with c2:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'emergency_stop': False, 'manual_fail': False, 'event_logs': []})
        st.rerun()
with c3:
    if st.button("💥 BRUCH SIMULIEREN", type="primary", use_container_width=True): st.session_state.manual_fail = True
with c4:
    st.subheader(f"Durchläufe: {st.session_state.count}")

# --- 5. SIMULATION ---
if st.session_state.is_run:
    st.session_state.count += 1
    age = min(2, st.session_state.count // 40)
    
    # Kausale Logik für Simulationstyp
    if st.session_state.manual_fail:
        true_z = 2
        st.session_state.manual_fail = False
    else:
        # Wahrscheinlichkeit für Defekt steigt bei Edelstahl + HSS
        base_risk = 0.95 if (werkst_sel == "Edelstahl" and bohrer_sel == "HSS") else 0.99
        true_z = 1 if (age >= 1 and np.random.random() > base_risk) else 0
    
    # Physikalische Werte generieren
    v_amp = np.random.normal(loc=(85 if true_z == 2 else 15), scale=8) 
    t_val = np.random.normal(loc=(78 if true_z == 1 else (45 if werkst_sel == "Edelstahl" else 28)), scale=4)
    torque = np.random.normal(loc=(65 if werkst_sel == "Edelstahl" else 30), scale=5)
    
    ev = {
        'Vibration': 1 if v_amp > 50 else 0, 'Strom': 1 if (true_z == 1 or np.random.random() < 0.05) else 0,
        'Temperatur': 1 if t_val > 55 else 0, 'Akustik': 1 if (true_z > 0 and np.random.random() > 0.4) else 0,
        'Drehmoment': 1 if torque > 45 else 0, 'Vorschub': 1 if (true_z == 2) else 0,
        'BohrerMat': 0 if bohrer_sel == "HSS" else 1, 'WerkstMat': 0 if werkst_sel == "Aluminium" else 1,
        'Alter': age
    }
    res = inf.query(['Zustand'], evidence=ev).values
    
    # Log-System
    timestamp = time.strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] #{st.session_state.count}: {bohrer_sel} in {werkst_sel} -> "
    if res[2] > 0.5: log_msg += "🔴 KRITISCH"
    elif res[1] > 0.5: log_msg += "🟡 STUMPF"
    else: log_msg += "🟢 OK"
    st.session_state.event_logs.insert(0, log_msg)
    
    st.session_state.history.append({'t': st.session_state.count, 'Intakt': res[0], 'Stumpf': res[1], 'Bruch': res[2], 'v_amp': v_amp, 'temp': t_val, 'torque': torque})
    
    if auto_stop_active and res[2] > 0.9:
        st.session_state.is_run = False
        st.session_state.emergency_stop = True
else:
    res = [1, 0, 0]
    ev = None

# --- 6. DISPLAY ---
if st.session_state.emergency_stop:
    st.error("🚨 NOT-HALT: Bruchgefahr bei über 90%! Bohrer gesichert.")

st.write("---")
col_tele, col_graph, col_log = st.columns([1, 2, 1.2])

with col_tele:
    st.subheader("📡 Echtzeit-Daten")
    l_v = st.session_state.history[-1]['v_amp'] if st.session_state.history else 0
    l_t = st.session_state.history[-1]['temp'] if st.session_state.history else 0
    l_d = st.session_state.history[-1]['torque'] if st.session_state.history else 0
    
    st.markdown(f'<div class="sensor-tile"><small>VIBRATION</small><br><b style="color:{"#ef4444" if l_v > 50 else "#3b82f6"}">{l_v:.1f} mm/s</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile"><small>TEMPERATUR</small><br><b style="color:{"#f97316" if l_t > 55 else "#10b981"}">{l_t:.1f} °C</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile"><small>DREHMOMENT</small><br><b>{l_d:.1f} Nm</b></div>', unsafe_allow_html=True)
    
    st.caption("Maschinen-Status")
    st.info(f"Setup: {bohrer_sel} / {werkst_sel}")

with col_brain:
    st.subheader("🧠 Bayessche Diagnose")
    b1, b2, b3 = st.columns(3)
    b1.metric("Intakt", f"{res[0]:.1%}")
    b2.metric("Stumpf", f"{res[1]:.1%}")
    b3.metric("Bruch", f"{res[2]:.1%}", delta=f"{res[2]*100:.1f}%", delta_color="inverse")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['Bruch'], name="Bruch", fill='tozeroy', line=dict(color='#ef4444')))
        fig.add_trace(go.Scatter(x=df['t'], y=df['Stumpf'], name="Verschleiß", line=dict(color='#f59e0b')))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.5)', height=250, margin=dict(l=0,r=0,t=0,b=0), font=dict(color="white"), xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

with col_log:
    st.subheader("📜 System-Logbuch")
    log_html = "".join([f'<div class="log-entry">{log}</div>' for log in st.session_state.event_logs])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

if ev:
    with st.expander("💡 Kausale Begründung", expanded=True):
        if bohrer_sel == "HSS" and werkst_sel == "Edelstahl":
            st.warning("Kritische Paarung: HSS-Bohrer in Edelstahl führt zu beschleunigter thermischer Degradation.")
        if ev['Temperatur'] and werkst_sel == "Edelstahl":
            st.write("Hohe Temperatur ist bei Edelstahl normal, aber in Kombination mit Drehmoment-Peaks steigt die Stumpf-Wahrscheinlichkeit.")
        if ev['Vibration']:
            st.write("Vibrations-Peaks detektiert. Das Modell gewichtet dies als primäres Indiz für einen Bruch.")

if st.session_state.is_run:
    time.sleep(speed)
    st.rerun()
