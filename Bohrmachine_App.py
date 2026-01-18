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

# --- 2. KOMPLEXES BAYESSCHES NETZWERK ---
@st.cache_resource
def create_material_bn(noise_v, bohrer_typ, werkst_typ):
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'), ('BohrerMat', 'Zustand'), ('WerkstMat', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Strom'),
        ('Zustand', 'Temperatur'), ('Zustand', 'Akustik'),
        ('Zustand', 'Drehmoment'), ('Zustand', 'Vorschub')
    ])
    
    cpd_a = TabularCPD('Alter', 3, [[0.7], [0.2], [0.1]]) 
    cpd_bm = TabularCPD('BohrerMat', 2, [[1.0 if bohrer_typ == "HSS" else 0.0], [1.0 if bohrer_typ == "Hartmetall" else 0.0]])
    cpd_wm = TabularCPD('WerkstMat', 2, [[1.0 if werkst_typ == "Aluminium" else 0.0], [1.0 if werkst_typ == "Edelstahl" else 0.0]])
    
    cpd_z = TabularCPD('Zustand', 3, [
            [0.99, 0.95, 0.80, 0.50, 0.90, 0.70, 0.40, 0.10, 0.70, 0.40, 0.10, 0.01], 
            [0.01, 0.04, 0.15, 0.30, 0.09, 0.20, 0.40, 0.50, 0.25, 0.40, 0.50, 0.40], 
            [0.00, 0.01, 0.05, 0.20, 0.01, 0.10, 0.20, 0.40, 0.05, 0.20, 0.40, 0.59]], 
            evidence=['Alter', 'BohrerMat', 'WerkstMat'], evidence_card=[3, 2, 2])
    
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
    bohrer_sel = st.selectbox("Bohrermaterial", ["HSS", "Hartmetall"])
    werkst_sel = st.selectbox("Werkstückmaterial", ["Aluminium", "Edelstahl"])
    st.divider()
    speed = st.select_slider("Simulations-Takt", [1.0, 0.5, 0.2, 0.1, 0.01], 0.1)
    auto_stop_active = st.checkbox("Not-Halt aktiv", value=True)

bn = create_material_bn(0.05, bohrer_sel, werkst_sel)
inf = VariableElimination(bn)

# --- 4. DASHBOARD HEADER ---
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
    if st.button("💥 BRUCH SIMULIEREN", type="primary", use_container_width=True): 
        st.session_state.manual_fail = True
with c4:
    st.subheader(f"Durchläufe: {st.session_state.count}")

# --- 5. SIMULATION LOGIK ---
if st.session_state.is_run:
    st.session_state.count += 1
    age = min(2, st.session_state.count // 40)
    
    if st.session_state.manual_fail:
        true_z = 2
        st.session_state.manual_fail = False
    else:
        risk = 0.94 if (werkst_sel == "Edelstahl" and bohrer_sel == "HSS") else 0.99
        true_z = 1 if (age >= 1 and np.random.random() > risk) else 0
    
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
    
    timestamp = time.strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] #{st.session_state.count}: {bohrer_sel}/{werkst_sel}"
    st.session_state.event_logs.insert(0, log_msg)
    
    st.session_state.history.append({'t': st.session_state.count, 'Intakt': res[0], 'Stumpf': res[1], 'Bruch': res[2], 'v_amp': v_amp, 'temp': t_val, 'torque': torque})
    
    if auto_stop_active and res[2] > 0.9:
        st.session_state.is_run = False
        st.session_state.emergency_stop = True
else:
    res = [1, 0, 0]
    ev = None

# --- 6. DISPLAY LAYOUT ---
if st.session_state.emergency_stop:
    st.error("🚨 NOT-HALT AUSGELÖST!")

st.write("---")
# Hier definieren wir die Spalten neu und nutzen sie sofort
col_a, col_b, col_c = st.columns([1, 2, 1.2])

with col_a:
    st.subheader("📡 Sensordaten")
    h = st.session_state.history[-1] if st.session_state.history else {'v_amp':0, 'temp':0, 'torque':0}
    st.markdown(f'<div class="sensor-tile">VIB: {h["v_amp"]:.1f} mm/s</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile">TEMP: {h["temp"]:.1f} °C</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sensor-tile">TORQ: {h["torque"]:.1f} Nm</div>', unsafe_allow_html=True)

with col_b:
    st.subheader("🧠 KI-Status")
    m1, m2, m3 = st.columns(3)
    m1.metric("Intakt", f"{res[0]:.0%}")
    m2.metric("Stumpf", f"{res[1]:.0%}")
    m3.metric("Bruch", f"{res[2]:.0%}")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['Bruch'], name="Bruch", fill='tozeroy', line_color='red'))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with col_c:
    st.subheader("📜 Logbuch")
    log_html = "".join([f'<div class="log-entry">{l}</div>' for l in st.session_state.event_logs[:50]])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

if st.session_state.is_run:
    time.sleep(speed)
    st.rerun()
