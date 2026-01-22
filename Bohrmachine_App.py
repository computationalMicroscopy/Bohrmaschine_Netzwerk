import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="KI-Expertensystem Bohrtechnik PRO", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; margin-bottom: 15px;
    }
    .val-title { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.2px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; }
    .emergency-alert {
        background: #f85149; color: white; padding: 15px; border-radius: 8px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 2px solid white; animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.5; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-LOGIK (MULTIVARIATE INFERENZ) ---
def calculate_risk_inference(alter, last, thermik, vibration, kss_ausfall, integritaet):
    # Logistische Regressions-Inferenz für die Bruchprognose
    z = (alter * 1.5) + (last * 2.8) + (thermik * 4.2) + (vibration * 3.5) + (kss_ausfall * 5.0) + ((100 - integritaet) * 0.12)
    risk = 1 / (1 + np.exp(-(z - 8.0)))
    return np.clip(risk, 0.001, 0.999)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42)
    }

MATERIALIEN = {
    "S235JR (Baustahl)": {"kc1.1": 1900, "mc": 0.26, "rate": 0.15, "t_crit": 450},
    "42CrMo4 (Stahl)": {"kc1.1": 2100, "mc": 0.25, "rate": 0.25, "t_crit": 550},
    "1.4404 (Edelstahl)": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650},
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
    sens_vibr = st.slider("Vibrations-Gain (Rauschen)", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Gain (Drehmoment)", 0.1, 5.0, 1.0)
    zyklus_sprung = st.number_input("Schrittweite [Zyklen]", 1, 100, 10)
    sim_takt = st.select_slider("Simulationstakt (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. PHYSIK- & KI-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    
    # Drehmoment-Berechnung (Zerspanungs-Last)
    kc = m['kc1.1'] * (f ** -m['mc'])
    drehmoment = (kc * f * (d/2) * (d/2)) / 1000 * sens_load 
    
    # Verschleiß & Thermik
    v_zuwachs = ((m['rate'] * (vc**1.7) * f) / (12000 if kss else 300)) * zyklus_sprung
    s['verschleiss'] += v_zuwachs
    t_ziel = 22 + (s['verschleiss'] * 1.4) + (vc * 0.22) + (0 if kss else 280)
    s['thermik'] += (t_ziel - s['thermik']) * 0.25
    
    # Vibration (Abhängig von Verschleiß und Last)
    vibr_base = (s['verschleiss'] * 0.08) + (vc * 0.015) + (drehmoment * 0.05)
    s['vibration'] = (vibr_base * sens_vibr) + s['seed'].normal(0, 0.25)
    s['vibration'] = max(0.1, s['vibration'])
    
    # KI-Bruchrisiko Inferenz
    s['risk'] = calculate_risk_inference(s['zyklus']/1000, drehmoment/60, s['thermik']/m['t_crit'], s['vibration']/10, 1.0 if not kss else 0.0, s['integritaet'])
    
    # Detaillierte XAI-Schadensberechnung
    loss_fatigue = (s['verschleiss'] / 100) * 0.04 * zyklus_sprung
    loss_load = (drehmoment / 100) * 0.01 * zyklus_sprung
    loss_thermal = (np.exp(max(0, s['thermik'] - m['t_crit']) / 45) - 1) * zyklus_sprung * 2
    loss_vibr = (s['vibration'] / 20) * 0.05 * zyklus_sprung
    
    total_loss = loss_fatigue + loss_load + loss_thermal + loss_vibr
    s['integritaet'] -= total_loss
    
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0

    log = {
        'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integritaet'], 
        'last': drehmoment, 'temp': s['thermik'], 'vib': s['vibration'],
        'f_loss': loss_fatigue, 'l_loss': loss_load, 't_loss': loss_thermal, 'v_loss': loss_vibr
    }
    s['logs'].insert(0, log)
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration'], 'l': drehmoment})

# --- 6. UI DASHBOARD ---
tab1, tab2 = st.tabs(["📊 LIVE-PROZESS-ANALYSE", "🧪 EXPERIMENTELLES SZENARIO-LABOR"])

with tab1:
    if s['broken']: st.markdown('<div class="emergency-alert">🚨 KRITISCHER AUSFALL: WERKZEUG-INTEGRITÄT BEI 0%</div>', unsafe_allow_html=True)
    
    # Metriken-Leiste
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.2f}%</span></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#e3b341">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="glass-card"><span class="val-title">Thermik</span><br><span class="val-main" style="color:#f85149">{s["thermik"]:.1f}°C</span></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><span class="val-main" style="color:#58a6ff">{s["vibration"]:.2f}</span></div>', unsafe_allow_html=True)
    with m5: st.markdown(f'<div class="glass-card"><span class="val-title">Last</span><br><span class="val-main" style="color:#bc8cff">{s["logs"][0]["last"]:.1f}Nm</span></div>' if s['logs'] else '---', unsafe_allow_html=True)

    col_graph, col_xai = st.columns([2.3, 1.7])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Struktur-Integrität [%]", "Thermik [°C]", "Vibration [mm/s]", "KI-Bruchrisiko [%]"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], name="Integrität", fill='tozeroy', line=dict(color='#3fb950', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], name="Thermik", line=dict(color='#f85149')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], name="Vibration", line=dict(color='#58a6ff')), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, name="Risiko", line=dict(color='#e3b341')), 4, 1)
            fig.update_layout(height=800, template="plotly_dark", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_xai:
        st.markdown("### 🔍 XAI-Deep-Monitor (Ursachen-Splitting)")
        x_html = ""
        for l in s['logs'][:12]:
            x_html += f"""
            <div style="border-left: 4px solid #e3b341; background: rgba(255,255,255,0.05); padding: 12px; margin-bottom: 10px; border-radius: 4px; font-size: 12px;">
                <div style="display:flex; justify-content:space-between;"><b>ZYKLUS-ZEIT: {l['zeit']}</b> <b>RISIKO: {l['risk']:.2%}</b></div>
                <div style="margin-top:8px;">
                    <span style="color:#f85149; font-weight:bold;">Integritäts-Verlust (Abzug pro Schritt):</span><br>
                    <table style="width:100%; margin-top:5px; font-family:monospace;">
                        <tr><td>• Ermüdung:</td><td style="text-align:right;">-{l['f_loss']:.5f}%</td></tr>
                        <tr><td>• Last-Stress:</td><td style="text-align:right;">-{l['l_loss']:.5f}%</td></tr>
                        <tr><td>• Thermik:</td><td style="text-align:right;">-{l['t_loss']:.5f}%</td></tr>
                        <tr><td>• Vibration:</td><td style="text-align:right;">-{l['v_loss']:.5f}%</td></tr>
                        <tr style="border-top:1px solid #777;"><td><b>SUMME:</b></td><td style="text-align:right;"><b>-{(l['f_loss']+l['l_loss']+l['t_loss']+l['v_loss']):.5f}%</b></td></tr>
                    </table>
                </div>
            </div>"""
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:780px; overflow-y:auto; padding-right:5px;">{x_html}</div>', height=800)

with tab2:
    st.header("🧪 Was-Wäre-Wenn Simulations-Labor")
    sc1, sc2, sc3 = st.columns([1.2, 1.2, 2])
    with sc1:
        st.subheader("Mechanische Last")
        sim_alter = st.slider("Werkzeug-Alter [Zyklen]", 0, 2000, 500)
        sim_last = st.slider("Zerspanungs-Last [Nm]", 0, 200, 40)
        sim_integ = st.slider("Vorschaden Integrität [%]", 0, 100, 100)
    with sc2:
        st.subheader("Umgebung & Sensorik")
        sim_temp = st.slider("Thermik-Simulation [°C]", 20, 1000, 150)
        sim_vibr = st.slider("Vibrations-Level [mm/s]", 0.0, 20.0, 2.0)
        sim_kss = st.toggle("KSS-Totalausfall simulieren", value=False)
    with sc3:
        r_sim = calculate_risk_inference(sim_alter/800, sim_last/50, sim_temp/500, sim_vibr/5, 1.0 if sim_kss else 0.0, sim_integ)
        st.markdown(f'<div class="glass-card" style="text-align:center;">Prognostiziertes Risikon<h1 style="color:#e3b341; font-size:5rem; margin:10px 0;">{r_sim:.2%}</h1></div>', unsafe_allow_html=True)
        
        # Impact Radar
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[sim_alter/20, sim_last/2, sim_temp/10, sim_vibr*5, (100 if sim_kss else 0)],
            theta=['Alter','Last','Thermik','Vibration','KSS-Fehler'], fill='toself', line=dict(color='#e3b341')
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.button("▶ START / STOPP SIMULATION", use_container_width=True): s['active'] = not s['active']
with col_b2:
    if st.button("🔄 VOLLSTÄNDIGER RESET (NEUES WERKZEUG)", use_container_width=True):
        st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42)}
        st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
