import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-END INDUSTRIAL STYLING ---
st.set_page_config(layout="wide", page_title="KI-Zerspanungs-Plattform TwinPro V5.2", page_icon="⚙️")

# --- INITIALISIERUNG & STATE-MACHINE ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'zyklus': 0.0, 'zyklen_anzahl': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }

s = st.session_state.twin

MATERIALIEN = {
    "Baustahl (1.0037)": {"kc1.1": 1800, "mc": 0.25, "wear_factor": 0.004, "t_crit": 450, "color": "#4a5568"}, 
    "Vergütungsstahl (1.7225)": {"kc1.1": 2100, "mc": 0.24, "wear_factor": 0.012, "t_crit": 550, "color": "#2d3748"}, 
    "Titanlegierung (3.7165)": {"kc1.1": 2500, "mc": 0.23, "wear_factor": 0.055, "t_crit": 650, "color": "#718096"},
    "Edelstahl (1.4301)": {"kc1.1": 2300, "mc": 0.22, "wear_factor": 0.038, "t_crit": 600, "color": "#a0aec0"}
}

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Live-Prozessparameter")
    mat_name = st.selectbox("Ausgewählter Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    
    vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 30, 350, 100)
    f = st.slider("Vorschub pro Umdrehung f [mm/U]", 0.05, 0.60, 0.15)
    d = st.slider("Bohrer-Durchmesser d [mm]", 5.0, 32.0, 12.0)
    kuehlung = st.toggle("Kühlschmierstoff (KSS) aktiv", value=True)
    
    st.divider()
    st.header("🎛️ Zeitskalierung & Sensoren")
    schrittweite = st.number_input("Zeitskalierungsfaktor", min_value=1, max_value=1000000, value=5)
    taktzeit = st.select_slider("Aktualisierungsintervall (ms)", options=[500, 200, 100, 0], value=100)
    
    sensor_temp_gain = st.slider("Temperatursensor-Empfindlichkeit", 0.5, 2.5, 1.0, step=0.1)
    sensor_vibr_gain = st.slider("Vibrationssensor-Verstärkung", 0.5, 3.0, 1.0, step=0.1)
    noise_level = st.slider("Rausch-Amplitude (Vibration)", 0.1, 2.0, 0.5)

# DYNAMISCHE ANIMATIONS-ZEITBERECHNUNG
basis_zyklus_zeit = 2.5 
dynamische_animations_dauer = max(0.05, basis_zyklus_zeit / schrittweite)

st.html(f"""
    <style>
    .stApp {{ background-color: #06090e; color: #c9d1d9; font-family: 'Inter', sans-serif; }}
    .main-title {{ font-size: 2.8rem; font-weight: 800; color: #f0f6fc; text-align: center; margin-bottom: 30px; background: linear-gradient(90deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .glass-card {{ background: rgba(16, 22, 30, 0.85); border: 1px solid rgba(48, 54, 61, 0.9); border-radius: 12px; padding: 20px; margin-bottom: 15px; text-align: center; }}
    .val-title {{ font-size: 1.05rem; color: #8b949e; text-transform: uppercase; font-weight: 700; }}
    .val-main {{ font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 800; color: #f0f6fc; }}
    @keyframes helical_spin {{ 100% {{ background-position: 0px -180px, 0px 0px; }} }}
    @keyframes tool_feed_optimized {{ 0%, 15%, 90%, 100% {{ transform: translateY(-35px); }} 25%, 65% {{ transform: translateY(22px); }} }}
    @keyframes led_pulse {{ 0%, 100% {{ box-shadow: 0 0 10px var(--led-color); }} 50% {{ box-shadow: 0 0 26px var(--led-color); }} }}
    </style>
""")

st.html('<div class="main-title">🚀 Next-Gen KI-Zerspanungslabor & XAI-Plattform</div>')

# --- DIAGNOSE ENGINE ---
def compute_sensor_diagnostics(current_vals, settings, integrity, kuehlung, m):
    M, T, V, F = current_vals['M'], current_vals['T'], current_vals['V'], current_vals['F']
    d = settings['d']
    crit_torque = 0.12 * (d ** 3)
    crit_force = 320 * (d ** 2)
    evidenz = {
        "Mechanische Torsions-Überlast": min(100.0, (M / crit_torque) * 100) if M > crit_torque * 0.8 else 0,
        "Thermische Gefüge-Erweichung": min(100.0, (T / m['t_crit']) * 100) if T > m['t_crit']*0.8 else 0,
        "Regeneratives Rattern": min(100.0, (V / 8.0) * 100) if V > 5.0 else 0,
        "Axiale Schaft-Knickung": min(100.0, (F / crit_force) * 100) if F > crit_force * 0.8 else 0
    }
    top_reason = max(evidenz, key=evidenz.get)
    mapping = {
        "Mechanische Torsions-Überlast": {"diag": "CRITICAL TORQUE", "exp": "Schaftquerschnitt überlastet.", "act": "Vorschub um 30% reduzieren!"},
        "Thermische Gefüge-Erweichung": {"diag": "THERMAL OVERLOAD", "exp": "Schneidkanten-Härteverlust.", "act": "vc um 25% drosseln!"},
        "Regeneratives Rattern": {"diag": "RESONANT CHATTER", "exp": "Resonanz stört Prozess.", "act": "Drehzahl leicht variieren!"},
        "Axiale Schaft-Knickung": {"diag": "AXIAL BUCKLED", "exp": "Stabilitätsversagen.", "act": "Vorschub sofort halbieren!"}
    }
    res = mapping.get(top_reason, {"diag": "NOMINAL", "exp": "Parameter im Sollbereich.", "act": "Keine Aktion erforderlich."})
    return res, sorted(evidenz.items(), key=lambda x: x[1], reverse=True)

# --- PHYSIK-ENGINE ---
n = (vc * 1000) / (np.pi * d) if d > 0 else 0
if s['active'] and not s['broken']:
    dt = (0.1) * schrittweite
    s['zyklus'] += dt
    kc = m['kc1.1'] * ((f/2.0) ** -m['mc'])
    s['drehmoment'] = (f * (d**2) * kc) / 8000.0
    s['vorschubkraft'] = (0.5 * d * f * kc) * 1.3
    s['thermik'] += (22.0 + (s['drehmoment'] * 0.1) - s['thermik']) * 0.2
    s['vibration'] = 0.2 + (s['drehmoment'] * 0.05)
    s['integritaet'] = max(0.0, s['integritaet'] - 0.01 * dt)
    exp, ev = compute_sensor_diagnostics({'M': s['drehmoment'], 'T': s['thermik'], 'V': s['vibration'], 'F': s['vorschubkraft']}, {'d': d}, s['integritaet'], kuehlung, m)
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'info': exp, 'evidenz': ev})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 't': s['thermik'], 'm': s['drehmoment'], 'f': s['vorschubkraft'], 'v': s['vibration'], 'p': s['leistung']})

# --- UI RENDERING MIT DYNAMISCHER SKALIERUNG ---
col_anim, col_met = st.columns([1.5, 4])

# DYNAMISCHE BERECHNUNG DER SKALIERUNG
scale = d / 12.0
dw = 44 * scale
dh = 115 * scale

with col_anim:
    st.markdown(f"""
        <div class="glass-card" style="min-height: 400px; display: flex; flex-direction: column; align-items: center;">
            <div style="animation: tool_feed_optimized {dynamische_animations_dauer}s infinite ease-in-out;">
                <div style="animation: helical_spin {max(0.01, 45/n if n>0 else 0):.3f}s linear infinite; 
                            width: {dw}px; height: {dh}px; 
                            background: repeating-linear-gradient(135deg, #12161b 0px, #3a3a3a 16px, #777 22px);">
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_met:
    c1, c2, c3 = st.columns(3)
    c1.html(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main">{s["integritaet"]:.1f}%</span></div>')
    c2.html(f'<div class="glass-card"><span class="val-title">Drehmoment</span><br><span class="val-main">{s["drehmoment"]:.1f} Nm</span></div>')
    c3.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main">{s["thermik"]:.0f} °C</span></div>')

# --- STEUERUNG ---
if st.button("▶ Start / Pause Simulation"): s['active'] = not s['active']
if st.button("🔄 Reset"): 
    st.session_state.twin = {'zyklus': 0.0, 'zyklen_anzahl': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False, 'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0, 'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0, 'seed': np.random.RandomState(42)}
    st.rerun()

# --- TABS & GRAPHS ---
t1, t2 = st.tabs(["📈 Echtzeit-Telemetrie", "🔬 Diagnose"])
with t1:
    if s['history']:
        df = pd.DataFrame(s['history'])
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=df['z'], y=df['i'], name="Integrität"), 1, 1)
        fig.add_trace(go.Scatter(x=df['z'], y=df['m'], name="Nm"), 2, 1)
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

if s['active']: 
    time.sleep(0.1)
    st.rerun()
