import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP UND DESIGN ---
st.set_page_config(layout="wide", page_title="KI-Präzisionsbohr-Labor v15", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .sensor-card { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
    .risk-high { color: #f85149 !important; }
    .log-area { font-family: 'Consolas', monospace; font-size: 0.85rem; height: 450px; overflow-y: auto; background: #010409; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KAUSALE KI-ENGINE (DEUTSCH) ---
@st.cache_resource
def build_expert_bn(n_v, n_t):
    # Zustand hängt ab von Verschleiß, Last und Kühlung
    model = DiscreteBayesianNetwork([
        ('Verschleiss', 'Zustand'), ('Last', 'Zustand'), ('Kuehlung', 'Zustand'),
        ('Zustand', 'Vibration'), ('Zustand', 'Temperatur'), ('Zustand', 'Stromaufnahme')
    ])
    
    # Prioren (Wahrscheinlichkeitsverteilung der Einflussfaktoren)
    cpd_v = TabularCPD('Verschleiss', 3, [[0.33], [0.33], [0.34]]) # Gering, Mittel, Hoch
    cpd_l = TabularCPD('Last', 2, [[0.7], [0.3]])                # Normal, Überlast
    cpd_k = TabularCPD('Kuehlung', 2, [[0.95], [0.05]])          # OK, Ausfall
    
    # Zustand CPT (Die Kernlogik des Bruchrisikos)
    z_matrix = []
    for v in range(3):
        for l in range(2):
            for k in range(2):
                risk_score = (v * 2.5) + (l * 4.0) + (k * 6.0)
                p_bruch = min(0.98, (risk_score**2.2) / 200.0) 
                p_warnung = min(1.0 - p_bruch, risk_score / 15.0)
                p_sicher = 1.0 - p_warnung - p_bruch
                z_matrix.append([p_sicher, p_warnung, p_bruch])
                
    cpd_zustand = TabularCPD('Zustand', 3, np.array(z_matrix).T, 
                            evidence=['Verschleiss', 'Last', 'Kuehlung'], evidence_card=[3, 2, 2])
    
    # Sensoren (Plausibilität der Messwerte)
    cpd_vib = TabularCPD('Vibration', 2, [[1-n_v, 0.4, 0.05], [n_v, 0.6, 0.95]], evidence=['Zustand'], evidence_card=[3])
    cpd_temp = TabularCPD('Temperatur', 2, [[0.98, 0.3, 0.01], [0.02, 0.7, 0.99]], evidence=['Zustand'], evidence_card=[3])
    cpd_strom = TabularCPD('Stromaufnahme', 2, [[0.9, 0.2, 0.1], [0.1, 0.8, 0.9]], evidence=['Zustand'], evidence_card=[3])
    
    model.add_cpds(cpd_v, cpd_l, cpd_k, cpd_zustand, cpd_vib, cpd_temp, cpd_strom)
    return model

# --- 3. SESSION STATE ---
if 'state' not in st.session_state:
    st.session_state.state = {
        'zyklus': 0, 'verschleiss': 0.0, 'historie': [], 'logs': [], 
        'aktiv': False, 'gebrochen': False, 'rng': np.random.RandomState(42)
    }

# --- 4. BEDIENPANEL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Maschinen-Steuerung")
    
    with st.expander("Werkzeug & Material", expanded=True):
        bohrer = st.selectbox("Werkzeugtyp", ["HSS-E (Beschichtet)", "VHM-Vollhartmetall"])
        material = st.selectbox("Werkstückstoff", ["Alu-Guss", "Edelstahl 1.4301", "Titan Grad 5"])
        vc = st.slider("Schnittgeschw. vc [m/min]", 10, 350, 100)
        f = st.slider("Vorschub f [mm/U]", 0.01, 0.6, 0.15)
    
    with st.expander("Sensor-Kalibrierung"):
        n_vib = st.slider("Vibrations-Rauschen", 0.0, 1.0, 0.1)
        n_temp = st.slider("Thermisches Rauschen", 0.0, 1.0, 0.05)
        instabil = st.slider("Aufspann-Instabilität", 0.0, 1.0, 0.1)
    
    kuehlung = st.toggle("Kühlung aktiv", value=True)
    takt = st.select_slider("Simulationstakt [ms]", [500, 200, 100, 50], 100)

# --- 5. SIMULATIONS-LOGIK ---
bn = build_expert_bn(n_vib, n_temp)
infer = VariableElimination(bn)

if st.session_state.state['aktiv'] and not st.session_state.state['gebrochen']:
    s = st.session_state.state
    s['zyklus'] += 1
    
    # Physikalische Verschleißberechnung (Angelehnt an Taylor)
    mat_faktor = {"Alu-Guss": 0.03, "Edelstahl 1.4301": 0.4, "Titan Grad 5": 2.5}[material]
    werkzeug_boni = 2.5 if "VHM" in bohrer else 0.8
    # vc hat exponentiellen Einfluss auf Verschleiß (typisch Zerspanung)
    last_index = (vc**1.5 * f * 5) / 1000
    
    v_zuwachs = (mat_faktor * last_index) / (werkzeug_boni * (15 if kuehlung else 0.5))
    # Stochastische Mikro-Ausbrüche
    if s['rng'].rand() > 0.98: v_zuwachs *= 4.0 
    
    s['verschleiss'] += v_zuwachs
    
    # Sensordaten generieren
    ist_ueberlast = 1 if (last_index > 1.5 or s['rng'].rand() > 0.96) else 0
    vib_ist = 12 + (s['verschleiss'] * 0.4) + (instabil * 45) + s['rng'].normal(0, 3)
    temp_ist = 22 + (s['verschleiss'] * 0.9) + (vc * 0.15) + (0 if kuehlung else 140) + s['rng'].normal(0, 2)
    
    # KI-Inferenz (Probabilistische Bewertung)
    beweis = {
        'Verschleiss': 0 if s['verschleiss'] < 30 else (1 if s['verschleiss'] < 80 else 2),
        'Last': ist_ueberlast,
        'Kuehlung': 0 if kuehlung else 1,
        'Vibration': 1 if vib_ist > 55 else 0,
        'Temperatur': 1 if temp_ist > 90 else 0
    }
    
    ergebnis = infer.query(['Zustand'], evidence=beweis).values
    risiko = ergebnis[2]
    
    # Bruch-Bedingungen
    if risiko > 0.97 or s['verschleiss'] > 140 or (risiko > 0.8 and s['rng'].rand() > 0.99):
        s['gebrochen'] = True
        s['aktiv'] = False
        
    # Datenaufzeichnung
    s['historie'].append({
        'z': s['zyklus'], 'r': risiko, 'v_grad': s['verschleiss'], 'vib': vib_ist, 'temp': temp_ist
    })
    s['logs'].insert(0, f"[{time.strftime('%H:%M:%S')}] ZYK {s['zyklus']:04d} | RISIKO: {risiko:.1%} | TEMP: {temp_ist:.1f}°C")

# --- 6. DASHBOARD AUSGABE ---
st.title("🔩 Digitaler Zwilling: KI-Zerspanungslabor")
st.caption("Professionelle Simulation von Werkzeugverschleiß und probabilistischer Zustandsüberwachung")

# Steuer-Buttons
c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1: 
    label = "⏸ PAUSE" if st.session_state.state['aktiv'] else "▶️ START"
    if st.button(label, use_container_width=True): st.session_state.state['aktiv'] = not st.session_state.state['aktiv']
with c2: 
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.state = {'zyklus':0,'verschleiss':0.0,'historie':[],'logs':[],'aktiv':False,'gebrochen':False,'rng':np.random.RandomState(42)}
        st.rerun()
with c3:
    if st.button("💥 BRUCH ERZWINGEN", use_container_width=True): st.session_state.state['verschleiss'] = 115

if st.session_state.state['gebrochen']:
    st.error(f"🚨 KRITISCHER AUSFALL: Werkzeugbruch in Zyklus {st.session_state.state['zyklus']}. Verschleißgrad: {st.session_state.state['verschleiss']:.1f}%")

st.divider()

# Live-Metriken
m1, m2, m3, m4 = st.columns(4)
aktuell = st.session_state.state['historie'][-1] if st.session_state.state['historie'] else {'r':0,'v_grad':0,'vib':0,'temp':0}

with m1: st.markdown(f'<div class="sensor-card">ZYKLUS<br><span class="metric-val">{st.session_state.state["zyklus"]}</span></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="sensor-card">VERSCHLEISS<br><span class="metric-val">{st.session_state.state["verschleiss"]:.1f}%</span></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="sensor-card">VIBRATION<br><span class="metric-val">{aktuell["vib"]:.1f}g</span></div>', unsafe_allow_html=True)
with m4:
    r_color = "risk-high" if aktuell['r'] > 0.7 else ""
    st.markdown(f'<div class="sensor-card">BRUCH-RISIKO<br><span class="metric-val {r_color}">{aktuell["r"]:.1%}</span></div>', unsafe_allow_html=True)

# Graphen
g1, g2 = st.columns([2, 1])

with g1:
    if st.session_state.state['historie']:
        df = pd.DataFrame(st.session_state.state['historie'])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, name="KI-Risiko-Index (%)", fill='tozeroy', line=dict(color='#f85149', width=3)))
        fig.add_trace(go.Scatter(x=df['z'], y=df['v_grad'], name="Mechanischer Verschleiß (%)", line=dict(color='#e3b341', dash='dot')), secondary_y=True)
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="KI-Risiko (%)", secondary_y=False)
        fig.update_yaxes(title_text="Verschleiß (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("🛠️ Prozess-Telemetrie")
    log_h = "".join([f"<div style='margin-bottom:4px; color:{'#f85149' if '9' in l[:20] else '#8b949e'}'>{l}</div>" for l in st.session_state.state['logs'][:50]])
    st.markdown(f'<div class="log-area">{log_h}</div>', unsafe_allow_html=True)

if st.session_state.state['aktiv']:
    time.sleep(takt/1000)
    st.rerun()
