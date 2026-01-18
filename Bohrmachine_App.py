import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & THEME ---
st.set_page_config(layout="wide", page_title="KI-Physik-Labor v18", page_icon="🔬")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .sensor-card { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 12px; padding: 15px; text-align: center;
    }
    .cycle-counter {
        font-family: 'JetBrains Mono', monospace; font-size: 3rem; 
        font-weight: 800; color: #79c0ff;
    }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
    .log-area { font-family: 'Consolas', monospace; font-size: 0.75rem; height: 350px; overflow-y: auto; background: #010409; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DAS ULTIMATIVE KI-MODELL (ALLE SENSOREN) ---
@st.cache_resource
def build_full_stack_bn(n_v, n_t, n_s):
    model = DiscreteBayesianNetwork([
        ('Zyklen_Alter', 'Zustand'), ('Verschleiss', 'Zustand'), ('Last', 'Zustand'), ('Kuehlung', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur'), ('Zustand', 'Stromaufnahme')
    ])
    
    cpd_zy = TabularCPD('Zyklen_Alter', 3, [[0.33], [0.33], [0.34]])
    cpd_ve = TabularCPD('Verschleiss', 3, [[0.33], [0.33], [0.34]])
    cpd_la = TabularCPD('Last', 2, [[0.7], [0.3]])
    cpd_kh = TabularCPD('Kuehlung', 2, [[0.95], [0.05]])
    
    # Zustand CPT
    z_matrix = []
    for zy in range(3):
        for ve in range(3):
            for la in range(2):
                for kh in range(2):
                    score = (zy * 3.0) + (ve * 2.5) + (la * 4.5) + (kh * 6.0)
                    p_bruch = min(0.99, (score**2.5) / 350.0)
                    p_warn = min(1.0 - p_bruch, score / 10.0)
                    z_matrix.append([1.0-p_warn-p_bruch, p_warn, p_bruch])
                    
    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Zyklen_Alter', 'Verschleiss', 'Last', 'Kuehlung'], 
                       evidence_card=[3, 3, 2, 2])
    
    cpd_vib = TabularCPD('Vibration', 2, [[1-n_v, 0.4, 0.05], [n_v, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_tmp = TabularCPD('Temperatur', 2, [[1-n_t, 0.3, 0.01], [n_t, 0.7, 0.99]], evidence=['Zustand'], evidence_card=[3])
    cpd_str = TabularCPD('Stromaufnahme', 2, [[1-n_s, 0.2, 0.1], [n_s, 0.8, 0.9]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_zy, cpd_ve, cpd_la, cpd_kh, cpd_z, cpd_vib, cpd_tmp, cpd_str)
    return model

# --- 3. SESSION STATE ---
if 'state' not in st.session_state:
    st.session_state.state = {
        'zyklus': 0, 'verschleiss': 0.0, 'historie': [], 'logs': [], 
        'aktiv': False, 'gebrochen': False, 'temp_akt': 20.0, 'rng': np.random.RandomState(42)
    }

# --- 4. BEDIENPANEL (ALLE REGLER ZURÜCK) ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    with st.expander("Prozess-Parameter", expanded=True):
        bohrer = st.selectbox("Bohrertyp", ["HSS-Co5", "VHM-TiAlN"])
        material = st.selectbox("Werkstoff", ["Alu", "Edelstahl 1.4301", "Titan Grad 5"])
        vc = st.slider("vc (Schnittgeschw.)", 10, 400, 120)
        f = st.slider("f (Vorschub)", 0.01, 0.8, 0.15)
        d_bohrer = st.number_input("Ø Bohrer [mm]", 1.0, 50.0, 10.0)
    
    with st.expander("Sensor- & Umgebungsrauschen"):
        n_vib = st.slider("Vibrations-Noise", 0.0, 1.0, 0.15)
        n_temp = st.slider("Thermal-Noise", 0.0, 1.0, 0.05)
        n_str = st.slider("Strom-Noise", 0.0, 1.0, 0.02)
        instabil = st.slider("Instabilität", 0.0, 1.0, 0.0)
    
    kuehlung = st.toggle("KSS-Kühlung aktiv", value=True)
    takt = st.select_slider("Sim-Takt [ms]", [500, 200, 100, 50], 100)

# --- 5. PHYSIK- & KI-LOGIK ---
bn = build_full_stack_bn(n_vib, n_temp, n_str)
infer = VariableElimination(bn)

if st.session_state.state['aktiv'] and not st.session_state.state['gebrochen']:
    s = st.session_state.state
    s['zyklus'] += 1
    
    # Physik-Berechnung
    kc = {"Alu": 800, "Edelstahl 1.4301": 2200, "Titan Grad 5": 3500}[material]
    md = ((d_bohrer / 2) * f * kc * d_bohrer) / 2000
    pwr = (md * (vc * 1000 / (np.pi * d_bohrer))) / 9550
    
    # Verschleiß & Temperatur
    v_mat = {"Alu": 0.02, "Edelstahl 1.4301": 0.3, "Titan Grad 5": 1.5}[material]
    s['verschleiss'] += (v_mat * (vc**1.4) * f) / (10000 if kuehlung else 400)
    
    ziel_t = 20 + (s['verschleiss'] * 1.2) + (vc * 0.1) + (0 if kuehlung else 160)
    s['temp_akt'] += (ziel_t - s['temp_akt']) * 0.2
    
    # Sensoren simulieren
    vib_ist = 10 + (s['verschleiss'] * 0.5) + (instabil * 50) + s['rng'].normal(0, 5)
    ampere = pwr * 4.5 + s['rng'].normal(0, 0.5)
    
    # KI-Inferenz
    zyk_idx = 0 if s['zyklus'] < 150 else (1 if s['zyklus'] < 500 else 2)
    ver_idx = 0 if s['verschleiss'] < 30 else (1 if s['verschleiss'] < 85 else 2)
    evidenz = {
        'Zyklen_Alter': zyk_idx, 'Verschleiss': ver_idx,
        'Last': 1 if md > (d_bohrer * 1.6) else 0, 'Kuehlung': 0 if kuehlung else 1,
        'Vibration': 1 if vib_ist > 60 else 0, 'Temperatur': 1 if s['temp_akt'] > 95 else 0
    }
    risiko = infer.query(['Zustand'], evidence=evidenz).values[2]
    
    if risiko > 0.98 or s['verschleiss'] > 145:
        s['gebrochen'] = True
        s['aktiv'] = False

    s['historie'].append({'z':s['zyklus'], 'r':risiko, 'w':s['verschleiss'], 't':s['temp_akt'], 'v':vib_ist, 'a':ampere, 'md':md})
    s['logs'].insert(0, f"Zyk {s['zyklus']}: Risiko {risiko:.1%} | {s['temp_akt']:.1f}°C | {vib_ist:.1f}g")

# --- 6. DASHBOARD ---
st.title("🔩 Pro-Physik-Labor: Zerspanung & KI v18")

# Top-Leiste: Zyklen & Risiko
c_top1, c_top2 = st.columns([1, 2])
with c_top1:
    st.markdown(f'<div class="sensor-card"><small>BOHRZYKLEN (KUMULATIV)</small><br><div class="cycle-counter">{st.session_state.state["zyklus"]}</div></div>', unsafe_allow_html=True)
with c_top2:
    if st.session_state.state['historie']:
        df = pd.DataFrame(st.session_state.state['historie'])
        fig_r = go.Figure(go.Scatter(x=df['z'], y=df['r']*100, fill='tozeroy', name="Bruchrisiko %", line=dict(color='#ff4b4b')))
        fig_r.update_layout(height=180, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_r, use_container_width=True)

# Steuerung
c_b1, c_b2, c_b3 = st.columns(3)
with c_b1: 
    if st.button("▶️ SIMULATION START/STOP", use_container_width=True): st.session_state.state['aktiv'] = not st.session_state.state['aktiv']
with c_b2:
    if st.button("🔄 LABOR-RESET", use_container_width=True):
        st.session_state.state = {'zyklus':0,'verschleiss':0.0,'historie':[],'logs':[],'aktiv':False,'gebrochen':False,'temp_akt':20.0,'rng':np.random.RandomState(42)}
        st.rerun()
with c_b3:
    if st.session_state.state['gebrochen']: st.error("WERKZEUGBRUCH: Standzeit überschritten!")

st.divider()

# Sensoren-Anzeige
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.state['historie'][-1] if st.session_state.state['historie'] else {'w':0,'v':0,'t':20,'a':0,'md':0}

with m1: st.markdown(f'<div class="sensor-card">Vibration<br><span class="metric-val">{last["v"]:.1f} g</span></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="sensor-card">Stromaufnahme<br><span class="metric-val">{last["a"]:.2f} A</span></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="sensor-card">Temperatur<br><span class="metric-val">{last["t"]:.1f} °C</span></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="sensor-card">Drehmoment<br><span class="metric-val">{last["md"]:.1f} Nm</span></div>', unsafe_allow_html=True)

# Hauptgrafik & Log
col_l, col_r = st.columns([2, 1])
with col_l:
    if st.session_state.state['historie']:
        df = pd.DataFrame(st.session_state.state['historie'])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['z'], y=df['v'], name="Vibration (g)", line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=df['z'], y=df['w'], name="Verschleiß (%)", line=dict(color='orange', dash='dot')), secondary_y=True)
        fig.update_layout(height=400, template="plotly_dark", title="Sensordynamik & Alterung")
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("📜 Sensor-Protokoll")
    log_h = "".join([f"<div style='border-bottom:1px solid #333'>{l}</div>" for l in st.session_state.state['logs'][:50]])
    st.markdown(f'<div class="log-area">{log_h}</div>', unsafe_allow_html=True)

if st.session_state.state['aktiv']:
    time.sleep(takt/1000)
    st.rerun()
