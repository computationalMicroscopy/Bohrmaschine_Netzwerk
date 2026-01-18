import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP ---
st.set_page_config(layout="wide", page_title="KI-Physik-Labor v16", page_icon="🔬")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .sensor-card { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 12px; padding: 15px; text-align: center;
    }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .log-area { font-family: 'Consolas', monospace; font-size: 0.8rem; height: 350px; overflow-y: auto; background: #010409; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-ENGINE (ERWEITERT) ---
@st.cache_resource
def build_physics_bn(n_v, n_t):
    model = DiscreteBayesianNetwork([
        ('Verschleiss', 'Zustand'), ('Drehmoment', 'Zustand'), ('Kuehlung', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur'), ('Zustand', 'Leistungsaufnahme')
    ])
    
    cpd_v = TabularCPD('Verschleiss', 3, [[0.33], [0.33], [0.34]])
    cpd_d = TabularCPD('Drehmoment', 2, [[0.8], [0.2]]) # Normal vs Spitzenlast
    cpd_k = TabularCPD('Kuehlung', 2, [[0.95], [0.05]])
    
    z_matrix = []
    for v in range(3):
        for d in range(2):
            for k in range(2):
                # Drehmoment ist ein kritischer Faktor für Torsionsbruch
                score = (v * 2.0) + (d * 5.0) + (k * 6.0)
                p_bruch = min(0.99, (score**2.5) / 250.0)
                p_warn = min(1.0 - p_bruch, score / 12.0)
                p_ok = 1.0 - p_warn - p_bruch
                z_matrix.append([p_ok, p_warn, p_bruch])
                
    cpd_z = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                       evidence=['Verschleiss', 'Drehmoment', 'Kuehlung'], evidence_card=[3, 2, 2])
    
    cpd_vib = TabularCPD('Vibration', 2, [[1-n_v, 0.4, 0.05], [n_v, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_temp = TabularCPD('Temperatur', 2, [[0.98, 0.3, 0.01], [0.02, 0.7, 0.99]], evidence=['Zustand'], evidence_card=[3])
    cpd_pwr = TabularCPD('Leistungsaufnahme', 2, [[0.9, 0.2, 0.1], [0.1, 0.8, 0.9]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_v, cpd_d, cpd_k, cpd_z, cpd_vib, cpd_temp, cpd_pwr)
    return model

# --- 3. SESSION STATE ---
if 'state' not in st.session_state:
    st.session_state.state = {
        'zyklus': 0, 'verschleiss': 0.0, 'historie': [], 'logs': [], 
        'aktiv': False, 'gebrochen': False, 'temp_aktuell': 20.0, 'rng': np.random.RandomState(42)
    }

# --- 4. BEDIENPANEL ---
with st.sidebar:
    st.header("⚙️ Maschinenparameter")
    
    with st.expander("Werkzeuggeometrie", expanded=True):
        d_bohrer = st.number_input("Bohrerdurchmesser [mm]", 1.0, 50.0, 10.0)
        bohrer_typ = st.selectbox("Beschichtung", ["TiN", "TiAlN (Hard)", "Diamond-Like Carbon"])
        material = st.selectbox("Werkstoff", ["Alu", "Stahl 42CrMo4", "Inconel 718"])
        
    with st.expander("Prozessdaten"):
        vc = st.slider("vc (m/min)", 10, 400, 120)
        f = st.slider("f (mm/U)", 0.01, 0.8, 0.12)
        kuehlung = st.toggle("Kühlschmierung (KSS)", value=True)

    with st.expander("Störgrößen"):
        n_vib = st.slider("Vibrationsrauschen", 0.0, 1.0, 0.1)
        st_takt = st.select_slider("Sim-Takt [ms]", [500, 200, 100, 50], 100)

# --- 5. PHYSIK-ENGINE ---
bn = build_physics_bn(n_vib, 0.05)
infer = VariableElimination(bn)

if st.session_state.state['aktiv'] and not st.session_state.state['gebrochen']:
    s = st.session_state.state
    s['zyklus'] += 1
    
    # Physikalische Berechnungen
    # Spezifische Schnittkraft kc (vereinfacht)
    kc = {"Alu": 800, "Stahl 42CrMo4": 2100, "Inconel 718": 3500}[material]
    
    # Schnittkraft Fc = b * h * kc
    fc = (d_bohrer / 2) * f * kc
    # Drehmoment Md = (Fc * D) / 2000 [Nm]
    md = (fc * d_bohrer) / 2000
    # Leistungsaufnahme P = Md * n / 9550
    drehzahl = (vc * 1000) / (np.pi * d_bohrer)
    leistung = (md * drehzahl) / 9550
    
    # Verschleißdynamik
    v_faktor = {"Alu": 0.02, "Stahl 42CrMo4": 0.3, "Inconel 718": 1.8}[material]
    v_zuwachs = (v_faktor * (vc**1.6) * f) / (15000 if kuehlung else 800)
    s['verschleiss'] += v_zuwachs
    
    # Thermodynamik (Abkühlkurve/Erhitzung)
    ziel_temp = 25 + (s['verschleiss'] * 0.8) + (vc * 0.12) + (0 if kuehlung else 180)
    s['temp_aktuell'] += (ziel_temp - s['temp_aktuell']) * 0.2 # Trägheit
    
    # KI-Inferenz
    beweis = {
        'Verschleiss': 0 if s['verschleiss'] < 30 else (1 if s['verschleiss'] < 85 else 2),
        'Drehmoment': 1 if md > (d_bohrer * 1.5) else 0, # Torsionsgefahr
        'Kuehlung': 0 if kuehlung else 1,
        'Vibration': 1 if s['rng'].rand() > 0.95 else 0,
        'Temperatur': 1 if s['temp_aktuell'] > 100 else 0
    }
    risiko = infer.query(['Zustand'], evidence=beweis).values[2]
    
    if risiko > 0.98 or s['verschleiss'] > 140:
        s['gebrochen'] = True
        s['aktiv'] = False

    s['historie'].append({'z': s['zyklus'], 'r': risiko, 'w': s['verschleiss'], 'md': md, 'p': leistung, 't': s['temp_aktuell']})
    s['logs'].insert(0, f"ZYK {s['zyklus']}: Md={md:.1f}Nm | P={leistung:.2f}kW | R={risiko:.1%}")

# --- 6. DASHBOARD ---
st.title("🛡️ KI-Physik-Labor: Zerspanung 4.0")

# Buttons
c1, c2, c3 = st.columns(3)
with c1: 
    if st.button("▶️ START/STOP", use_container_width=True): st.session_state.state['aktiv'] = not st.session_state.state['aktiv']
with c2:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.state = {'zyklus':0,'verschleiss':0.0,'historie':[],'logs':[],'aktiv':False,'gebrochen':False,'temp_aktuell':20.0,'rng':np.random.RandomState(42)}
        st.rerun()
with c3:
    if st.session_state.state['gebrochen']: st.error("WERKZEUGBRUCH!")

st.divider()

# Anzeigen
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.state['historie'][-1] if st.session_state.state['historie'] else {'md':0,'p':0,'t':20,'r':0}

with m1: st.markdown(f'<div class="sensor-card">Drehmoment<br><span class="metric-val">{last["md"]:.1f} Nm</span></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="sensor-card">Leistung<br><span class="metric-val">{last["p"]:.2f} kW</span></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="sensor-card">Temperatur<br><span class="metric-val">{last["t"]:.1f} °C</span></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="sensor-card">Bruch-Risiko<br><span class="metric-val" style="color:#f85149">{last["r"]:.1%}</span></div>', unsafe_allow_html=True)



# Grafiken
col_l, col_r = st.columns([2, 1])
with col_l:
    if st.session_state.state['historie']:
        df = pd.DataFrame(st.session_state.state['historie'])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, name="KI-Risiko (%)", fill='tozeroy', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['z'], y=df['md'], name="Drehmoment (Nm)", line=dict(color='cyan')), secondary_y=True)
        fig.update_layout(height=400, template="plotly_dark", title="KI-Risiko vs. Mechanisches Drehmoment")
        st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("📜 System-Log")
    log_h = "".join([f"<div>{l}</div>" for l in st.session_state.state['logs'][:50]])
    st.markdown(f'<div class="log-area">{log_h}</div>', unsafe_allow_html=True)

if st.session_state.state['aktiv']:
    time.sleep(st_takt/1000)
    st.rerun()
