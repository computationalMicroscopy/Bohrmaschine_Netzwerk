import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & DESIGN ---
st.set_page_config(layout="wide", page_title="KI - Labor Bohrtechnik ULTIMATE PRO", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px; padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px); margin-bottom: 15px;
    }
    .val-title { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 800; margin: 0; }
    .emergency-alert {
        background: #f85149; color: white; padding: 20px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 4px solid #ffffff; animation: blinker 0.8s linear infinite;
        font-size: 1.5rem;
    }
    @keyframes blinker { 50% { opacity: 0.3; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-LOGIK (KONTINUIERLICH) ---
def get_continuous_risk(age_norm, load_norm, therm_norm, cool_val, health_norm):
    # Sigmoid-Inferenz-Funktion
    score = (age_norm * 1.5) + (load_norm * 3.0) + (therm_norm * 4.5) + (cool_val * 4.0) + ((1.0 - health_norm) * 6.0)
    risk = 1 / (1 + np.exp(-(score - 4.5))) 
    return np.clip(risk, 0.01, 0.99)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 450},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.25, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.45, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.2, "temp_crit": 750}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Prozess-Eingriff")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    
    st.divider()
    st.header("📡 Sensorik & KI")
    sens_vib = st.slider("Vibrations-Gain", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.1, 5.0, 1.0)
    cycle_step = st.number_input("Schrittweite [Zyklen]", 1, 100, 10)
    sim_speed = st.select_slider("Simulations-Takt (ms)", options=[1000, 500, 200, 100, 50, 0], value=100)

# --- 5. LOGIK ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['cycle'] += cycle_step
    
    # 1. Kraft & Last (Physik)
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    
    # 2. Verschleiß & Temperatur
    wear_inc = ((mat['wear_rate'] * (vc ** 1.8) * f) / (12000 if cooling else 250)) * cycle_step
    s['wear'] += wear_inc
    target_t = 22 + (s['wear'] * 1.3) + (vc * 0.25) + (0 if cooling else 300)
    s['t_current'] += (target_t - s['t_current']) * 0.3
    
    # 3. Vibration
    s['vib'] = ((s['wear'] * 0.06) + (vc * 0.015) + (100 - s['integrity']) * 0.25) * sens_vib + s['seed'].normal(0, 0.3)
    s['vib'] = max(0.1, s['vib'])

    # 4. KI-Inferenz (Normalisiertes Risiko)
    s['risk'] = get_continuous_risk(
        age_norm = s['cycle']/1000, 
        load_norm = (mc_raw * sens_load)/60, 
        therm_norm = s['t_current']/mat['temp_crit'], 
        cool_val = 1.0 if not cooling else 0.0, 
        health_norm = s['integrity']/100
    )
    
    # 5. Struktur-Schädigung (Die "Wahrheit" hinter dem Risiko)
    f_loss = (s['wear'] / 100) * 0.04 * cycle_step # Ermüdung
    a_loss = (s['risk'] ** 3.0) * 0.8 * cycle_step if s['risk'] > 0.2 else 0 # Akut-Last
    t_loss = (np.exp(max(0, s['t_current'] - mat['temp_crit']) / 40) - 1) * cycle_step * 2 # Thermik
    
    s['integrity'] -= (f_loss + a_loss + t_loss)
    if s['integrity'] <= 0:
        s['broken'], s['active'], s['integrity'] = True, False, 0

    log_entry = {
        'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integrity'],
        'f_loss': f_loss, 'a_loss': a_loss, 't_loss': t_loss,
        'temp': s['t_current'], 'vib': s['vib'], 'mc': mc_raw, 'wear': s['wear']
    }
    s['logs'].insert(0, log_entry)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'i': s['integrity'], 't': s['t_current'], 'v': s['vib'], 'w': s['wear']})

# --- 6. UI ---
tab1, tab2 = st.tabs(["📊 LIVE-DASHBOARD", "🧪 SZENARIO-LABOR"])

with tab1:
    if s['broken']: st.markdown('<div class="emergency-alert">🚨 SYSTEM-CRASH: WERKZEUG GEBROCHEN</div>', unsafe_allow_html=True)
    
    # Metriken Reihe
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="glass-card"><span class="val-title">Struktur</span><br><p class="val-main" style="color:#3fb950">{s["integrity"]:.1f}%</p></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="glass-card"><span class="val-title">KI-Risiko</span><br><p class="val-main" style="color:#e3b341">{s["risk"]:.1%}</p></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><p class="val-main" style="color:#f85149">{s["t_current"]:.1f}°C</p></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><p class="val-main" style="color:#58a6ff">{s["vib"]:.2f}</p></div>', unsafe_allow_html=True)

    col_gra, col_xai = st.columns([2.2, 1.8])
    
    with col_gra:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Integrität", "Temperatur", "Vibrationen", "KI-Risiko %"))
            fig.add_trace(go.Scatter(x=df['c'], y=df['i'], name="Integrität", fill='tozeroy', line=dict(color='#3fb950')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['t'], name="Temp", line=dict(color='#f85149')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['v'], name="Vib", line=dict(color='#58a6ff')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="Risiko", line=dict(color='#e3b341')), row=4, col=1)
            fig.update_layout(height=700, template="plotly_dark", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_xai:
        st.markdown("### 🔍 XAI Deep-Monitor")
        x_html = ""
        for l in s['logs'][:12]:
            x_html += f"""
            <div style="border-left: 4px solid {"#f85149" if l['risk']>0.6 else "#3fb950"}; background: rgba(255,255,255,0.03); padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                <div style="display:flex; justify-content:space-between; font-size:12px;">
                    <b style="color:#58a6ff;">ZEIT: {l['zeit']}</b>
                    <b style="color:#e3b341;">RISIKO: {l['risk']:.1%}</b>
                </div>
                <div style="margin-top:8px; font-size:13px;">
                    <b>Physik-Status:</b> {l['temp']:.1f}°C | {l['vib']:.2f} mm/s | M_d: {l['mc']:.1f} Nm<br>
                    <span style="color:#f85149; font-weight:bold;">Integritäts-Verlust:</span><br>
                    <div style="padding-left:10px; font-family: monospace;">
                    • Fatigue-Stress: -{l['f_loss']:.4f}%<br>
                    • Akutlast-KI: -{l['a_loss']:.4f}%<br>
                    • Thermik-Gefüge: -{l['t_loss']:.4f}%
                    </div>
                </div>
            </div>"""
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:680px; overflow-y:auto;">{x_html}</div>', height=700)

with tab2:
    st.header("🧪 Experimentelles Labor")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        la_age = st.slider("Alter (Zyklen)", 0, 1500, 200)
        la_h = st.slider("Integrität (%)", 0, 100, 100)
    with c2:
        la_f = st.slider("Vorschub (mm/U)", 0.0, 1.2, 0.2)
        la_t = st.slider("Temperatur (°C)", 20, 900, 100)
        la_c = st.toggle("Keine Kühlung", value=False)
    with c3:
        risk_lab = get_continuous_risk(la_age/1000, la_f*3, la_t/500, 1.0 if la_c else 0.0, la_h/100)
        st.markdown(f'<div class="glass-card" style="text-align:center;"><h3>KI-Einschätzung</h3><h1 style="color:#e3b341; font-size:4rem;">{risk_lab:.1%}</h1></div>', unsafe_allow_html=True)
        if risk_lab > 0.8: st.error("Dringender Wartungsbedarf! Zusammenbruch steht bevor.")
        elif risk_lab > 0.4: st.warning("Erhöhte Degradation. Parameter anpassen!")
        else: st.success("Sicherer Prozessparameter.")

st.divider()
b1, b2 = st.columns(2)
with b1:
    if st.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
with b2:
    if st.button("🔄 VOLLSTÄNDIGER SYSTEM-RESET", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if s['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
