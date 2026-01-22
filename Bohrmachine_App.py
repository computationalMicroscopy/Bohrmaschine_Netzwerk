import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="KI-Labor Bohrertechnik", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .main-title {
        font-size: 2.5rem; font-weight: 800; color: #e3b341;
        margin-bottom: 20px; text-align: center; border-bottom: 2px solid #e3b341; padding-bottom: 10px;
    }
    .glass-card {
        background: rgba(23, 28, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; margin-bottom: 15px;
    }
    .val-title { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.2px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; }
    .xai-card {
        background: rgba(30, 35, 45, 0.9); border-left: 5px solid #e3b341;
        padding: 15px; border-radius: 8px; margin-bottom: 12px;
    }
    .reason-text { color: #e1e4e8; font-size: 0.85rem; margin-top: 5px; line-height: 1.4; font-style: italic; }
    .action-text { color: #58a6ff; font-weight: bold; font-size: 0.85rem; margin-top: 5px; }
    .emergency-alert {
        background: #f85149; color: white; padding: 15px; border-radius: 8px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">KI-Labor Bohrertechnik</div>', unsafe_allow_html=True)

# --- 2. LOGIK-FUNKTIONEN ---
def get_explanation(top_reason):
    mapping = {
        "Material-Ermüdung": ("Die KI erkennt Mikrorisse im Werkzeugstahl.", "Tausche das Werkzeug bald aus."),
        "Überlastung": ("Drehmoment zu hoch für diesen Durchmesser.", "Senke den Vorschub (f)."),
        "Gefüge-Überhitzung": ("Stahl erreicht Anlasstemperatur und wird weich.", "Kühlung prüfen oder vc senken."),
        "Resonanz-Instabilität": ("Schwingungen hämmern gegen die Schneidkante.", "Drehzahl leicht variieren."),
        "Kühlungs-Defizit": ("Extreme Reibung – Späne könnten verschweißen.", "KSS-Zufuhr sofort prüfen!"),
        "Struktur-Vorschaden": ("Vorschaden schwächt die Gesamtstabilität.", "Vorsicht: Bruchgefahr erhöht.")
    }
    return mapping.get(top_reason, ("Prozess stabil.", "Keine Aktion nötig."))

def calculate_metrics(alter, last, thermik, vibration, kss_ausfall, integritaet):
    w = [1.2, 2.4, 3.8, 3.0, 4.5, 0.10]
    scores = [alter * w[0], last * w[1], thermik * w[2], vibration * w[3], kss_ausfall * w[4], (100 - integritaet) * w[5]]
    z = sum(scores)
    risk = 1 / (1 + np.exp(-(z - 9.5)))
    labels = ["Material-Ermüdung", "Überlastung", "Gefüge-Überhitzung", "Resonanz-Instabilität", "Kühlungs-Defizit", "Struktur-Vorschaden"]
    evidenz = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    rul = int(max(0, (integritaet - 10) / max(0.01, (risk * 0.45))) * 5.5) if risk < 0.98 else 0
    return np.clip(risk, 0.001, 0.999), evidenz, rul

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}

MATERIALIEN = {
    "Baustahl": {"kc1.1": 1900, "mc": 0.26, "rate": 0.15, "t_crit": 450},
    "Vergütungsstahl": {"kc1.1": 2100, "mc": 0.25, "rate": 0.25, "t_crit": 550},
    "Edelstahl": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650},
    "Titan Grade 5": {"kc1.1": 2800, "mc": 0.24, "rate": 1.2, "t_crit": 750}
}

# --- 4. SIDEBAR (SENSOREN WIEDER DA) ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 20, 600, 180)
    f = st.slider("Vorschub f [mm/U]", 0.01, 1.2, 0.2)
    d = st.number_input("Bohrer-Durchmesser [mm]", 1.0, 100.0, 12.0)
    kss = st.toggle("Kühlschmierung aktiv", value=True)
    st.divider()
    st.header("📡 Sensor-Feineinstellung")
    sens_vibr = st.slider("Vibrations-Gain", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.1, 5.0, 1.0)
    st.divider()
    zyklus_sprung = st.number_input("Schrittweite [Zyklen]", 1, 100, 10)
    sim_takt = st.select_slider("Takt (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. PHYSIK-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    s['drehmoment'] = ((m['kc1.1'] * (f ** -m['mc']) * f * (d/2)**2) / 1000) * sens_load
    s['verschleiss'] += ((m['rate'] * (vc**1.7) * f) / (12000 if kss else 300)) * zyklus_sprung
    s['thermik'] += ((22 + (s['verschleiss']*1.4) + (vc*0.22) + (0 if kss else 280)) - s['thermik']) * 0.25
    s['vibration'] = ((s['verschleiss']*0.08 + vc*0.015 + s['drehmoment']*0.05) * sens_vibr) + s['seed'].normal(0, 0.2)
    s['risk'], evidenz_list, s['rul'] = calculate_metrics(s['zyklus']/1000, s['drehmoment']/60, s['thermik']/m['t_crit'], s['vibration']/10, 1.0 if not kss else 0.0, s['integritaet'])
    s['integritaet'] -= ((s['verschleiss']/100)*0.04 + (s['drehmoment']/100)*0.01 + (np.exp(max(0, s['thermik']-m['t_crit'])/45)-1)*2 + (s['vibration']/20)*0.05) * zyklus_sprung
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0
    exp, act = get_explanation(evidenz_list[0][0])
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'exp': exp, 'act': act})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration']})

# --- 6. UI ---
if s['broken']: st.markdown('<div class="emergency-alert">🚨 SYSTEM-STOPP: WERKZEUGBRUCH</div>', unsafe_allow_html=True)

# Kennzahlen
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.1f}%</span></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="glass-card"><span class="val-title">Risiko</span><br><span class="val-main" style="color:#e3b341">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="glass-card"><span class="val-title">Wartung in</span><br><span class="val-main" style="color:#58a6ff">{s["rul"]} Z.</span></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="glass-card"><span class="val-title">Thermik</span><br><span class="val-main" style="color:#f85149">{s["thermik"]:.0f}°C</span></div>', unsafe_allow_html=True)
m5.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><span class="val-main" style="color:#bc8cff">{max(0,s["vibration"]):.1f}</span></div>', unsafe_allow_html=True)
m6.markdown(f'<div class="glass-card"><span class="val-title">Last</span><br><span class="val-main">{s["drehmoment"]:.1f}Nm</span></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 LIVE-ANALYSE", "🧪 SZENARIO-LABOR (WAS-WÄRE-WENN)"])

with tab1:
    col_l, col_r = st.columns([2.2, 1.8])
    with col_l:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Historie: Integrität", "Sensorik: Hitze & Vibration", "KI: Bruchrisiko %"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], name="Integrität", fill='tozeroy', line=dict(color='#3fb950', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], name="Temp", line=dict(color='#f85149')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], name="Vibr", line=dict(color='#bc8cff')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, name="Risiko", line=dict(color='#e3b341', width=3)), 3, 1)
            fig.update_layout(height=650, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown("### 👨‍🏫 KI-Zustandsbericht")
        for l in s['logs'][:6]:
            st.markdown(f'<div class="xai-card"><b style="color:#e3b341;">{l["zeit"]} | Risiko {l["risk"]:.1%}</b><br><div class="reason-text">"{l["exp"]}"</div><div class="action-text">👉 {l["act"]}</div></div>', unsafe_allow_html=True)

with tab2:
    st.header("🧪 Experimentelles Labor")
    sc1, sc2, sc3 = st.columns([1, 1, 1.5])
    with sc1:
        sim_alter = st.slider("Alter [Zyklen]", 0, 2000, 500)
        sim_last = st.slider("Last [Nm]", 0, 200, 40)
    with sc2:
        sim_temp = st.slider("Hitze [°C]", 20, 1000, 150)
        sim_vibr = st.slider("Vibration", 0.0, 20.0, 2.0)
    with sc3:
        r_sim, evidenz_sim, rul_sim = calculate_metrics(sim_alter/800, sim_last/50, sim_temp/500, sim_vibr/5, 0.0, 100)
        st.markdown(f'<div class="glass-card" style="text-align:center;">Prognostizierter Wartungswert<h1>{rul_sim} Zyklen</h1><small>Bruchrisiko: {r_sim:.1%}</small></div>', unsafe_allow_html=True)

st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
if c2.button("🔄 NEUES WERKZEUG", use_container_width=True):
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
