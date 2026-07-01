import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP ---
st.set_page_config(layout="wide", page_title="KI-Labor Bohrertechnik", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .main-title { font-size: 2.5rem; font-weight: 800; color: #e3b341; margin-bottom: 20px; text-align: center; border-bottom: 2px solid #e3b341; padding-bottom: 10px; }
    .glass-card { background: rgba(23, 28, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 15px; }
    .val-title { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.2px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; }
    .xai-container { height: 650px; overflow-y: auto; padding-right: 10px; }
    .emergency-alert { background: #f85149; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOGIK & PHYSIK ---
def calculate_metrics_bayesian(prior_risk, alter, last, thermik, vibration, kuehlung_ausfall, integritaet):
    w = [1.2, 2.4, 3.8, 4.2, 4.5, 0.10]
    raw_scores = np.array([alter * w[0], last * w[1], thermik * w[2], (vibration/10) * w[3], kuehlung_ausfall * w[4], (100 - integritaet) * w[5]])
    exp_scores = np.exp(raw_scores * 0.8)
    probabilities = (exp_scores / exp_scores.sum()) * 100
    z = sum(raw_scores)
    likelihood = 1 / (1 + np.exp(-(z - 9.5)))
    posterior = (likelihood * 0.3) + (prior_risk * 0.7)
    labels = ["Material-Ermüdung", "Überlastung", "Gefüge-Überhitzung", "Resonanz-Instabilität", "Kühlungs-Defizit", "Struktur-Vorschaden"]
    evidenz = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
    rul = int(max(0, (integritaet - 10) / max(0.01, (posterior * 0.45))) * 5.5) if posterior < 0.98 else 0
    return np.clip(posterior, 0.001, 0.999), evidenz, rul

def get_drill_animation_html(vc, f, temp, vibration):
    # Animation steuert sich über Parameter
    rot_duration = max(0.1, 2.0 - (vc / 300))
    # Farbe ändert sich von Stahlgrau zu Glühend Rot
    glow_intensity = min(255, int((temp / 800) * 200))
    drill_color = f"rgb({150 + glow_intensity}, {150 - glow_intensity}, {150 - glow_intensity})"
    
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 250px; background: rgba(0,0,0,0.2); border-radius: 10px;">
        <svg width="200" height="200" viewBox="0 0 100 100">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>
            <rect x="45" y="0" width="10" height="100" fill="#333" />
            <g style="transform-origin: 50px 50px; animation: rotate {rot_duration}s linear infinite;">
                <rect x="40" y="30" width="20" height="50" fill="{drill_color}" rx="5" filter="url(#glow)"/>
                <path d="M40 30 L60 30 L50 80 Z" fill="#222" opacity="0.3"/>
            </g>
        </svg>
        <style>
            @keyframes rotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        </style>
    </div>
    """

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.01, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'drehmoment': 0.0}

MATERIALIEN = {"Baustahl": {"kc1.1": 1900, "mc": 0.26, "rate": 0.15, "t_crit": 450}, "Edelstahl": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650}}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 600, 180)
    f = st.slider("f [mm/U]", 0.01, 1.2, 0.2)
    d = st.number_input("Ø [mm]", 1.0, 100.0, 12.0)
    kuehlung = st.toggle("Kühlung aktiv", value=True)
    sim_takt = st.select_slider("Simulationstakt (ms)", options=[500, 100, 20], value=100)

# --- 5. PHYSIK-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += 1
    s['drehmoment'] = ((m['kc1.1'] * (f ** -m['mc']) * f * (d/2)**2) / 1000)
    s['verschleiss'] += ((m['rate'] * (vc**1.2) * f) / (12000 if kuehlung else 300))
    s['thermik'] += ((22 + (s['verschleiss']*1.4) + (vc*0.1) + (0 if kuehlung else 280)) - s['thermik']) * 0.1
    s['vibration'] = (s['verschleiss']*0.02 + vc*0.002 + s['drehmoment']*0.01) + np.random.normal(0, 0.2)
    
    s['risk'], evidenz_list, s['rul'] = calculate_metrics_bayesian(s['risk'], s['zyklus']/1000, s['drehmoment']/60, s['thermik']/m['t_crit'], s['vibration'], 1.0 if not kuehlung else 0.0, s['integritaet'])
    s['integritaet'] -= (max(0, s['thermik'] - m['t_crit']) * 0.01 + s['vibration'] * 0.1)
    
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 't': s['thermik'], 'v': s['vibration'], 'r': s['risk']})

# --- 6. UI ---
st.markdown('<div class="main-title">KI-Labor Bohrertechnik</div>', unsafe_allow_html=True)

if s['broken']: st.markdown('<div class="emergency-alert">🚨 SYSTEM-STOPP: WERKZEUGBRUCH</div>', unsafe_allow_html=True)

# Dashboard Metriken
m0, m1, m2, m3, m4 = st.columns(5)
m0.markdown(f'<div class="glass-card"><span class="val-title">Zyklen</span><br><span class="val-main">{s["zyklus"]}</span></div>', unsafe_allow_html=True)
m1.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.1f}%</span></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#e3b341">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:#f85149">{s["thermik"]:.0f}°C</span></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><span class="val-main" style="color:#bc8cff">{max(0,s["vibration"]):.1f}</span></div>', unsafe_allow_html=True)

# Layout
c_anim, c_graphs = st.columns([1, 2])

with c_anim:
    st.subheader("Live-Animation")
    st.markdown(get_drill_animation_html(vc, f, s['thermik'], s['vibration']), unsafe_allow_html=True)
    if st.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.01, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'drehmoment': 0.0}
        st.rerun()

with c_graphs:
    if s['history']:
        df = pd.DataFrame(s['history'])
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Sensor-Trends (Temp & Vibration)", "Integritätsverlauf"))
        fig.add_trace(go.Scatter(x=df['z'], y=df['t'], name="Temp", line=dict(color='#f85149')), 1, 1)
        fig.add_trace(go.Scatter(x=df['z'], y=df['v'], name="Vibration", line=dict(color='#bc8cff')), 1, 1)
        fig.add_trace(go.Scatter(x=df['z'], y=df['i'], name="Integrität", line=dict(color='#3fb950')), 2, 1)
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
