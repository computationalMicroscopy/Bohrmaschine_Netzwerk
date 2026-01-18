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
st.set_page_config(layout="wide", page_title="KI-Physik-Labor v17", page_icon="🔬")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .sensor-card { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 12px; padding: 15px; text-align: center;
    }
    .cycle-counter {
        font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; 
        font-weight: 800; color: #79c0ff; text-shadow: 0 0 10px #58a6ff66;
    }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .log-area { font-family: 'Consolas', monospace; font-size: 0.8rem; height: 350px; overflow-y: auto; background: #010409; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ERWEITERTES KI-MODELL (MIT ZYKLEN-ABHÄNGIGKEIT) ---
@st.cache_resource
def build_advanced_bn(n_v, n_t):
    # 'Zyklen' ist nun ein direkter Einflussfaktor auf den 'Zustand'
    model = DiscreteBayesianNetwork([
        ('Zyklen_Status', 'Zustand'), ('Verschleiss', 'Zustand'), 
        ('Drehmoment', 'Zustand'), ('Kuehlung', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur')
    ])
    
    # Prioren
    cpd_zy = TabularCPD('Zyklen_Status', 3, [[0.33], [0.33], [0.34]]) # Neu, Gebraucht, Alt
    cpd_ve = TabularCPD('Verschleiss', 3, [[0.33], [0.33], [0.34]])
    cpd_dm = TabularCPD('Drehmoment', 2, [[0.8], [0.2]])
    cpd_kh = TabularCPD('Kuehlung', 2, [[0.95], [0.05]])
    
    # Zustand CPT: 3(Zyk) * 3(Ver) * 2(Dreh) * 2(Kühl) = 36 Kombinationen
    z_matrix = []
    for zy in range(3): # Zyklen-Einfluss
        for ve in range(3):
            for dm in range(2):
                for kh in range(2):
                    # Zyklen erhöhen das Risiko massiv (Ermüdung)
                    score = (zy * 3.5) + (ve * 2.0) + (dm * 4.0) + (kh * 6.0)
                    p_bruch = min(0.995, (score**2.8) / 450.0)
                    p_warn = min(1.0 - p_bruch, score / 10.0)
                    p_ok = 1.0 - p_warn - p_bruch
                    z_matrix.append([p_ok, p_warn, p_bruch])
                    
    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Zyklen_Status', 'Verschleiss', 'Drehmoment', 'Kuehlung'], 
                       evidence_card=[3, 3, 2, 2])
    
    cpd_vib = TabularCPD('Vibration', 2, [[1-n_v, 0.4, 0.05], [n_v, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_tmp = TabularCPD('Temperatur', 2, [[0.98, 0.3, 0.01], [0.02, 0.7, 0.99]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_zy, cpd_ve, cpd_dm, cpd_kh, cpd_z, cpd_vib, cpd_tmp)
    return model

# --- 3. SESSION STATE ---
if 'state' not in st.session_state:
    st.session_state.state = {
        'zyklus': 0, 'verschleiss': 0.0, 'historie': [], 'logs': [], 
        'aktiv': False, 'gebrochen': False, 'temp_aktuell': 20.0, 'rng': np.random.RandomState(42)
    }

# --- 4. SEITENLEISTE ---
with st.sidebar:
    st.header("⚙️ Maschinenparameter")
    material = st.selectbox("Werkstoff", ["Alu", "Stahl 42CrMo4", "Inconel 718"])
    vc = st.slider("vc (m/min)", 10, 400, 100)
    f = st.slider("f (mm/U)", 0.01, 0.8, 0.10)
    d_bohrer = st.number_input("Bohrerdurchmesser [mm]", 1.0, 50.0, 8.0)
    kuehlung = st.toggle("Kühlung aktiv", value=True)
    st_takt = st.select_slider("Sim-Takt [ms]", [500, 200, 100, 50], 100)

# --- 5. PHYSIK-RECHNER ---
bn = build_advanced_bn(0.1, 0.05)
infer = VariableElimination(bn)

if st.session_state.state['aktiv'] and not st.session_state.state['gebrochen']:
    s = st.session_state.state
    s['zyklus'] += 1
    
    # Berechnung Drehmoment Md
    kc = {"Alu": 800, "Stahl 42CrMo4": 2100, "Inconel 718": 3500}[material]
    md = ((d_bohrer / 2) * f * kc * d_bohrer) / 2000
    
    # Verschleiß-Berechnung
    v_faktor = {"Alu": 0.01, "Stahl 42CrMo4": 0.2, "Inconel 718": 1.2}[material]
    s['verschleiss'] += (v_faktor * (vc**1.5) * f) / (12000 if kuehlung else 500)
    
    # Temperatur-Modell
    ziel_temp = 22 + (s['verschleiss'] * 1.1) + (vc * 0.1) + (0 if kuehlung else 150)
    s['temp_aktuell'] += (ziel_temp - s['temp_aktuell']) * 0.15
    
    # KI-INFERENZ mit Zyklen-Logik
    # Wir definieren Zyklen-Status: 0: <100, 1: 100-500, 2: >500 (Beispielhaft skaliert)
    zyk_idx = 0 if s['zyklus'] < 200 else (1 if s['zyklus'] < 600 else 2)
    ver_idx = 0 if s['verschleiss'] < 30 else (1 if s['verschleiss'] < 80 else 2)
    
    beweis = {
        'Zyklen_Status': zyk_idx,
        'Verschleiss': ver_idx,
        'Drehmoment': 1 if md > (d_bohrer * 1.8) else 0,
        'Kuehlung': 0 if kuehlung else 1
    }
    risiko = infer.query(['Zustand'], evidence=beweis).values[2]
    
    # Bruch-Logik
    if risiko > 0.98 or s['verschleiss'] > 150:
        s['gebrochen'] = True
        s['aktiv'] = False

    s['historie'].append({'z': s['zyklus'], 'r': risiko, 'w': s['verschleiss'], 't': s['temp_aktuell'], 'md': md})
    s['logs'].insert(0, f"Zyk {s['zyklus']}: Risiko {risiko:.1%} | Md {md:.1f}Nm")

# --- 6. DASHBOARD ---
st.title("🛡️ KI-Physik-Labor: Ermüdung & Bruchvorhersage")

# ZYKLEN-GROSSANZEIGE
c_top1, c_top2 = st.columns([1, 2])
with c_top1:
    st.markdown(f"""
        <div class="sensor-card">
            <small>GESAMTZYKLEN</small><br>
            <div class="cycle-counter">{st.session_state.state['zyklus']}</div>
        </div>
    """, unsafe_allow_html=True)
with c_top2:
    if st.session_state.state['historie']:
        df = pd.DataFrame(st.session_state.state['historie'])
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df['z'], y=df['r']*100, fill='tozeroy', name="Bruch-Risiko %", line=dict(color='#f85149')))
        fig_r.update_layout(height=180, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_r, use_container_width=True)

st.write("")

# Steuerung
c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
with c_ctrl1: 
    if st.button("▶️ START/STOP", use_container_width=True): st.session_state.state['aktiv'] = not st.session_state.state['aktiv']
with c_ctrl2:
    if st.button("🔄 SYSTEM-RESET", use_container_width=True):
        st.session_state.state = {'zyklus':0,'verschleiss':0.0,'historie':[],'logs':[],'aktiv':False,'gebrochen':False,'temp_aktuell':20.0,'rng':np.random.RandomState(42)}
        st.rerun()
with c_ctrl3:
    if st.session_state.state['gebrochen']: st.error("WERKZEUGBRUCH ERFOLGT!")

st.divider()

# Telemetrie
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.state['historie'][-1] if st.session_state.state['historie'] else {'md':0,'w':0,'t':20,'r':0}

with m1: st.markdown(f'<div class="sensor-card">Verschleiß<br><span class="metric-val">{last["w"]:.1f} %</span></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="sensor-card">Drehmoment<br><span class="metric-val">{last["md"]:.1f} Nm</span></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="sensor-card">Temperatur<br><span class="metric-val">{last["t"]:.1f} °C</span></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="sensor-card">KI-Konfidenz<br><span class="metric-val" style="color:{"#f85149" if last["r"] > 0.8 else "#58a6ff"}">{(1-last["r"]):.1%}</span></div>', unsafe_allow_html=True)

# Graphen & Log
col_graph, col_log = st.columns([2, 1])
with col_graph:
    if st.session_state.state['historie']:
        df = pd.DataFrame(st.session_state.state['historie'])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['z'], y=df['w'], name="Verschleiß", line=dict(color='#e3b341')))
        fig.add_trace(go.Scatter(x=df['z'], y=df['t'], name="Temperatur", line=dict(color='#ff7b72')), secondary_y=True)
        fig.update_layout(height=400, template="plotly_dark", title="Physikalische Trends")
        st.plotly_chart(fig, use_container_width=True)

with col_log:
    st.subheader("📜 Event-Log")
    log_h = "".join([f"<div>{l}</div>" for l in st.session_state.state['logs'][:50]])
    st.markdown(f'<div class="log-area">{log_h}</div>', unsafe_allow_html=True)

if st.session_state.state['aktiv']:
    time.sleep(st_takt/1000)
    st.rerun()
