import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. PRO-UI SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v13", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #0b0f14; color: #e0e0e0; }
    .sensor-tile { background-color: #161b22; border-radius: 10px; padding: 20px; border: 1px solid #30363d; text-align: center; }
    .metric-value { font-family: 'IBM Plex Mono', monospace; color: #58a6ff; font-size: 1.8rem; font-weight: bold; }
    .log-container { height: 400px; overflow-y: scroll; background-color: #0d1117; border: 1px solid #30363d; padding: 10px; font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DAS PERFEKTE MODELL (Realistische Wahrscheinlichkeiten) ---
@st.cache_resource
def create_final_bn(n_v, n_t, n_s, bohrer_mat, werkst_mat):
    model = DiscreteBayesianNetwork([
        ('Verschleiss', 'Zustand'), ('Kuehlung', 'Zustand'), ('Vorschub_Regler', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur'), ('Zustand', 'Vorschub_Ist')
    ])
    
    # Priors (Die Basis-Zustände)
    cpd_vschleiss = TabularCPD('Verschleiss', 3, [[0.33], [0.33], [0.34]])
    cpd_k = TabularCPD('Kuehlung', 2, [[0.9], [0.1]]) 
    cpd_vr = TabularCPD('Vorschub_Regler', 2, [[0.5], [0.5]])

    # Zustand CPT (Bruch-Logik)
    # 3 (Verschleiss) * 2 (Kühlung) * 2 (Vorschub) = 12 Spalten
    # Materialeinfluss wird hier direkt in die Wahrscheinlichkeit gewebt
    mat_risk = 1.5 if werkst_mat == "Titan" else (1.2 if werkst_mat == "Edelstahl" else 0.8)
    
    z_matrix = []
    for v in range(3): # 0=Neu, 1=Stumpf, 2=Kritisch
        for k in range(2): # 0=OK, 1=Ausfall
            for vr in range(2): # 0=Low, 1=High
                prob_bruch = (v * 0.25) + (k * 0.4) + (vr * 0.1)
                prob_bruch = min(0.95, prob_bruch * mat_risk)
                prob_stumpf = (v * 0.4) + (k * 0.2)
                prob_stumpf = min(1.0 - prob_bruch, prob_stumpf)
                prob_intakt = 1.0 - prob_bruch - prob_stumpf
                z_matrix.append([prob_intakt, prob_stumpf, prob_bruch])

    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Verschleiss', 'Kuehlung', 'Vorschub_Regler'], 
                       evidence_card=[3, 2, 2])
    
    # Sensoren (Sehr sensitiv auf Zustand)
    cpd_vib = TabularCPD('Vibration', 2, [[0.9, 0.4, 0.05], [0.1, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_temp = TabularCPD('Temperatur', 2, [[0.95, 0.3, 0.02], [0.05, 0.7, 0.98]], evidence=['Zustand'], evidence_card=[3])
    cpd_vi = TabularCPD('Vorschub_Ist', 2, [[0.99, 0.5, 0.01], [0.01, 0.5, 0.99]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_vschleiss, cpd_k, cpd_vr, cpd_z, cpd_vib, cpd_temp, cpd_vi)
    return model

# --- 3. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.update({
        'count': 0, 'history': [], 'is_run': False, 'logs': [], 
        'cum_wear': 0.0, 'has_broken': False, 'manual_fail': False
    })

# --- 4. SIDEBAR (Alle Regler) ---
with st.sidebar:
    st.header("🛠️ Maschinen-Konfiguration")
    b_mat = st.selectbox("Werkzeug-Typ", ["HSS-Bohrer", "VHM-Bohrer (Hartmetall)"])
    w_mat = st.selectbox("Material", ["Alu", "Edelstahl", "Titan"])
    
    st.divider()
    st.subheader("Prozess-Parameter")
    v_cut = st.slider("v_c (Schnittgeschwindigkeit)", 20, 250, 80)
    f_in = st.slider("f (Vorschub mm/U)", 0.05, 0.8, 0.15)
    
    st.divider()
    with st.expander("Sensor-Feineinstellung"):
        n_v = st.slider("Vibrations-Rauschen", 0.0, 1.0, 0.1)
        instability = st.slider("Instabile Aufspannung", 0.0, 1.0, 0.0)
    
    k_fail = st.toggle("🚨 Kühlmittelausfall simulieren")
    st_speed = st.select_slider("Sim-Geschwindigkeit", [1000, 500, 100, 10], 100)

# --- 5. LOGIK-KERN ---
bn = create_final_bn(n_v, 0.05, 0.02, b_mat, w_mat)
inf = VariableElimination(bn)

if st.session_state.is_run and not st.session_state.has_broken:
    st.session_state.count += 1
    
    # Realistischer Verschleiß-Zuwachs
    mat_mult = {"Alu": 0.05, "Edelstahl": 0.5, "Titan": 2.5}[w_mat]
    wear_inc = (v_cut * f_in * mat_mult * (10 if k_fail else 1)) / 500
    st.session_state.cum_wear += wear_inc
    
    # Sensoren simulieren
    vib_val = np.random.normal(15, 3) + (st.session_state.cum_wear * 0.6) + (instability * 40)
    temp_val = np.random.normal(35, 2) + (st.session_state.cum_wear * 0.5) + (120 if k_fail else 0)
    
    # Evidenz für KI
    ev = {
        'Verschleiss': 0 if st.session_state.cum_wear < 30 else (1 if st.session_state.cum_wear < 80 else 2),
        'Kuehlung': 1 if k_fail else 0,
        'Vorschub_Regler': 1 if f_in > 0.3 else 0,
        'Vibration': 1 if vib_val > 50 else 0,
        'Temperatur': 1 if temp_val > 70 else 0
    }
    
    res = inf.query(['Zustand'], evidence=ev).values
    prob_bruch = res[2]
    
    # Bruch-Event
    if prob_bruch > 0.95 or st.session_state.cum_wear > 120 or st.session_state.manual_fail:
        st.session_state.has_broken = True
        st.session_state.is_run = False
    
    st.session_state.history.append({'t': st.session_state.count, 'prob': prob_bruch, 'wear': st.session_state.cum_wear, 'temp': temp_val, 'vib': vib_val})
    st.session_state.logs.insert(0, f"Zyklus {st.session_state.count:04d}: Risiko {prob_bruch:.1%} | Verschleiß {st.session_state.cum_wear:.1f}%")

# --- 6. VISUALISIERUNG ---
st.title("🔩 Industrial AI Lab: Precision Drilling v13")

# Aktions-Buttons
c_b1, c_b2, c_b3, c_b4 = st.columns([1,1,1,2])
with c_b1:
    if st.button("▶️ START/STOP", use_container_width=True): st.session_state.is_run = not st.session_state.is_run
with c_b2:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.update({'count':0, 'history':[], 'is_run':False, 'logs':[], 'cum_wear':0.0, 'has_broken':False, 'manual_fail':False})
        st.rerun()
with c_b3:
    if st.button("💥 BRUCH", type="primary", use_container_width=True): st.session_state.manual_fail = True
with c_b4:
    if st.session_state.has_broken: st.error(f"SYSTEM STOPP: Werkzeugbruch in Zyklus {st.session_state.count}!")

st.write("---")

# Telemetrie-Karten
t1, t2, t3, t4 = st.columns(4)
with t1: st.markdown(f'<div class="sensor-tile"><small>ZYKLEN</small><br><span class="metric-value">{st.session_state.count}</span></div>', unsafe_allow_html=True)
with t2: st.markdown(f'<div class="sensor-tile"><small>VERSCHLEISS</small><br><span class="metric-value">{st.session_state.cum_wear:.1f}%</span></div>', unsafe_allow_html=True)
with t3: 
    v_val = st.session_state.history[-1]['vib'] if st.session_state.history else 0
    st.markdown(f'<div class="sensor-tile"><small>VIBRATION (g)</small><br><span class="metric-value">{v_val:.1f}</span></div>', unsafe_allow_html=True)
with t4:
    t_val = st.session_state.history[-1]['temp'] if st.session_state.history else 0
    st.markdown(f'<div class="sensor-tile"><small>TEMP (°C)</small><br><span class="metric-value">{t_val:.1f}</span></div>', unsafe_allow_html=True)

# Graphen
col_main, col_side = st.columns([2, 1])

with col_main:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['t'], y=df['prob']*100, name="Bruch-Risiko (%)", line=dict(color='#ff4b4b', width=4), fill='tozeroy'))
        fig.add_trace(go.Scatter(x=df['t'], y=df['wear'], name="Verschleiß (%)", line=dict(color='#eeb01f', dash='dot')))
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.subheader("📡 Inferenz-Log")
    log_h = "".join([f'<div style="border-bottom:1px solid #30363d; padding:3px;">{l}</div>' for l in st.session_state.logs[:50]])
    st.markdown(f'<div class="log-container">{log_h}</div>', unsafe_allow_html=True)

# Gauge für aktuelles Risiko
if st.session_state.history:
    curr_risk = st.session_state.history[-1]['prob'] * 100
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = curr_risk, title = {'text': "KI-Gefahren-Index"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#ff4b4b"}, 'steps': [{'range': [0, 50], 'color': "green"}, {'range': [50, 80], 'color': "yellow"}, {'range': [80, 100], 'color': "red"}]}
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

if st.session_state.is_run:
    time.sleep(st_speed/1000)
    st.rerun()
