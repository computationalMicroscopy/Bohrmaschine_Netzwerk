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
        margin-bottom: 20px; text-align: center; border-bottom: 2px solid #e3b341;
        padding-bottom: 10px;
    }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; margin-bottom: 15px;
    }
    .val-title { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.2px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; }
    .evidenz-tag {
        font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; 
        background: rgba(227, 179, 65, 0.15); color: #e3b341; 
        border: 1px solid #e3b341; margin-right: 5px; font-weight: bold;
    }
    .xai-label { color: #8b949e; font-size: 0.75rem; vertical-align: middle; }
    .xai-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #f85149; text-align: right; }
    .emergency-alert {
        background: #f85149; color: white; padding: 15px; border-radius: 8px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 2px solid white; animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.5; } }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">KI-Labor Bohrertechnik</div>', unsafe_allow_html=True)

# --- 2. KI-LOGIK (MODIFIZIERTE LABELS) ---
def calculate_metrics(alter, last, thermik, vibration, kss_ausfall, integritaet):
    w = [1.2, 2.4, 3.8, 3.0, 4.5, 0.10]
    scores = [alter * w[0], last * w[1], thermik * w[2], vibration * w[3], kss_ausfall * w[4], (100 - integritaet) * w[5]]
    z = sum(scores)
    risk = 1 / (1 + np.exp(-(z - 9.5)))
    
    # Präzisere Labels für die Evidenz-Tags
    labels = ["Material-Ermüdung", "Überlastung", "Gefüge-Überhitzung", "Resonanz-Instabilität", "Kühlungs-Defizit", "Struktur-Vorschaden"]
    evidenz = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    
    if risk > 0.98 or integritaet <= 0: zyklen_bis_wartung = 0
    else:
        verbleibend = max(0, (integritaet - 10) / max(0.01, (risk * 0.45)))
        zyklen_bis_wartung = int(verbleibend * 5.5)
        
    return np.clip(risk, 0.001, 0.999), evidenz, zyklen_bis_wartung

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42),
        'rul': 800, 'drehmoment': 0.0
    }

MATERIALIEN = {
    "Baustahl": {"kc1.1": 1900, "mc": 0.26, "rate": 0.15, "t_crit": 450},
    "Vergütungsstahl": {"kc1.1": 2100, "mc": 0.25, "rate": 0.25, "t_crit": 550},
    "Edelstahl": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650},
    "Titan Grade 5": {"kc1.1": 2800, "mc": 0.24, "rate": 1.2, "t_crit": 750}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Prozess-Konfiguration")
    mat_name = st.selectbox("Werkstoff-Auswahl", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 20, 600, 180)
    f = st.slider("Vorschub f [mm/U]", 0.01, 1.2, 0.2)
    d = st.number_input("Bohrer-Durchmesser [mm]", 1.0, 100.0, 12.0)
    kss = st.toggle("Kühlschmierung (KSS) aktiv", value=True)
    st.divider()
    st.header("📡 Sensor-Setup")
    sens_vibr = st.slider("Vibrations-Gain", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.1, 5.0, 1.0)
    zyklus_sprung = st.number_input("Schrittweite [Zyklen]", 1, 100, 10)
    sim_takt = st.select_slider("Simulationstakt (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. PHYSIK- & KI-ENGINE ---
s = st.session_state.twin

if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    kc = m['kc1.1'] * (f ** -m['mc'])
    s['drehmoment'] = (kc * f * (d/2) * (d/2)) / 1000 * sens_load 
    v_zuwachs = ((m['rate'] * (vc**1.7) * f) / (12000 if kss else 300)) * zyklus_sprung
    s['verschleiss'] += v_zuwachs
    t_ziel = 22 + (s['verschleiss'] * 1.4) + (vc * 0.22) + (0 if kss else 280)
    s['thermik'] += (t_ziel - s['thermik']) * 0.25
    vibr_base = (s['verschleiss'] * 0.08) + (vc * 0.015) + (s['drehmoment'] * 0.05)
    s['vibration'] = (vibr_base * sens_vibr) + s['seed'].normal(0, 0.25)
    s['vibration'] = max(0.1, s['vibration'])
    s['risk'], evidenz_list, s['rul'] = calculate_metrics(s['zyklus']/1000, s['drehmoment']/60, s['thermik']/m['t_crit'], s['vibration']/10, 1.0 if not kss else 0.0, s['integritaet'])
    
    l_fatigue = (s['verschleiss'] / 100) * 0.04 * zyklus_sprung
    l_load = (s['drehmoment'] / 100) * 0.01 * zyklus_sprung
    l_thermal = (np.exp(max(0, s['thermik'] - m['t_crit']) / 45) - 1) * zyklus_sprung * 2
    l_vibr = (s['vibration'] / 20) * 0.05 * zyklus_sprung
    
    s['integritaet'] -= (l_fatigue + l_load + l_thermal + l_vibr)
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0

    log = {
        'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integritaet'], 'rul': s['rul'],
        'l_fatigue': l_fatigue, 'l_load': l_load, 'l_thermal': l_thermal, 'l_vibr': l_vibr,
        'evidenz': evidenz_list[:3]
    }
    s['logs'].insert(0, log)
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration'], 'rul': s['rul']})

# --- 6. UI DASHBOARD ---
tab1, tab2 = st.tabs(["📊 LIVE-PROZESS-ANALYSE", "🧪 SZENARIO-LABOR"])

with tab1:
    if s['broken']: st.markdown('<div class="emergency-alert">🚨 SYSTEM-STOPP: STRUKTURVERSAGEN DETEKTIERT</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.1f}%</span></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#e3b341">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="glass-card"><span class="val-title">Wartung in...</span><br><span class="val-main" style="color:#58a6ff">{s["rul"]} Z.</span></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="glass-card"><span class="val-title">Thermik</span><br><span class="val-main" style="color:#f85149">{s["thermik"]:.1f}°C</span></div>', unsafe_allow_html=True)
    with m5: st.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><span class="val-main" style="color:#bc8cff">{s["vibration"]:.2f}</span></div>', unsafe_allow_html=True)
    with m6: st.markdown(f'<div class="glass-card"><span class="val-title">Zerspanlast</span><br><span class="val-main" style="color:#ffffff">{s["drehmoment"]:.1f}Nm</span></div>', unsafe_allow_html=True)

    col_graph, col_xai = st.columns([2.3, 1.7])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Integrität & Prognose", "Thermik [°C]", "Vibration [mm/s]", "KI-Bruchrisiko [%]"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#3fb950', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['rul'], line=dict(color='#58a6ff', dash='dot')), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#f85149')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#bc8cff')), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, line=dict(color='#e3b341')), 4, 1)
            fig.update_layout(height=800, template="plotly_dark", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_xai:
        st.markdown("### 🔍 KI-Entscheidungsbegründung")
        x_html = ""
        for l in s['logs'][:10]:
            badges = "".join([f'<span class="evidenz-tag">{e[0]}</span>' for e in l['evidenz']])
            x_html += f"""
            <div style="border-left: 5px solid #e3b341; background: rgba(30, 35, 45, 0.8); padding: 15px; margin-bottom: 12px; border-radius: 8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <b style="color:#58a6ff; font-size:13px;">ANALYSE: {l['zeit']}</b> 
                    <b style="color:#e3b341; font-size:13px;">RISIKO: {l['risk']:.1%}</b>
                </div>
                <div style="margin-bottom:10px;">{badges}</div>
                <div style="border-top: 1px solid #30363d; padding-top:8px;">
                    <p style="font-size:11px; color:#8b949e; margin-bottom:8px; font-weight:bold;">KAUSALE DEGRADATIONS-FAKTOREN:</p>
                    <div style="display: grid; grid-template-columns: 1fr 100px; font-size:11px; row-gap: 4px;">
                        <span class="xai-label">Mikrorisse durch Lastwechsel (Alter)</span><span class="xai-value">-{l['l_fatigue']:.4f}%</span>
                        <span class="xai-label">Mechanische Spannungsspitzen (Kraft)</span><span class="xai-value">-{l['l_load']:.4f}%</span>
                        <span class="xai-label">Thermische Enthärtung (Hitze)</span><span class="xai-value">-{l['l_thermal']:.4f}%</span>
                        <span class="xai-label">Kinetische Zerrüttung (Vibration)</span><span class="xai-value">-{l['l_vibr']:.4f}%</span>
                        <hr style="grid-column: span 2; border: 0.5px solid #444; margin: 4px 0;">
                        <span style="font-weight:bold; color:#e1e4e8;">Totaler Substanzverlust:</span>
                        <span style="font-weight:bold; color:#f85149; text-align:right;">-{(l['l_fatigue']+l['l_load']+l['l_thermal']+l['l_vibr']):.4f}%</span>
                    </div>
                </div>
            </div>"""
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:780px; overflow-y:auto; padding-right:5px;">{x_html}</div>', height=800)

with tab2:
    st.header("🧪 Experimentelles Simulations-Labor")
    sc1, sc2, sc3 = st.columns([1.2, 1.2, 2])
    with sc1:
        sim_alter = st.slider("Werkzeug-Alter [Zyklen]", 0, 2000, 500)
        sim_last = st.slider("Zerspanungs-Last [Nm]", 0, 200, 40)
        sim_integ = st.slider("Vorschaden Integrität [%]", 0, 100, 100)
    with sc2:
        sim_temp = st.slider("Thermik-Simulation [°C]", 20, 1000, 150)
        sim_vibr = st.slider("Vibrations-Level [mm/s]", 0.0, 20.0, 2.0)
        sim_kss = st.toggle("KSS-Totalausfall simulieren", value=False)
    with sc3:
        r_sim, evidenz_sim, rul_sim = calculate_metrics(sim_alter/800, sim_last/50, sim_temp/500, sim_vibr/5, 1.0 if sim_kss else 0.0, sim_integ)
        st.markdown(f'<div class="glass-card" style="text-align:center;">Wartungs-Empfehlung<h1 style="color:#58a6ff; font-size:5rem; margin:10px 0;">{rul_sim} Z.</h1><small>Statistisches Bruchrisiko: {r_sim:.1%}</small></div>', unsafe_allow_html=True)
        fig_radar = go.Figure(data=go.Scatterpolar(r=[sim_alter/20, sim_last/2, sim_temp/10, sim_vibr*5, (100 if sim_kss else 0)], theta=['Alter','Last','Thermik','Vibration','KSS-Fehler'], fill='toself', line=dict(color='#e3b341')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
if st.button("▶ SIMULATION START / STOPP", use_container_width=True): 
    s['active'] = not s['active']
if st.button("🔄 SYSTEM-RESET (NEUES WERKZEUG)", use_container_width=True):
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
