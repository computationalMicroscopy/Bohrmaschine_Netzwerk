import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & DESIGN ---
st.set_page_config(layout="wide", page_title="KI-Expertensystem Bohrtechnik v3.0", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; margin-bottom: 15px;
    }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; color: #e3b341; }
    .emergency-alert {
        background: #f85149; color: white; padding: 15px; border-radius: 8px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 2px solid white; animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.5; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-LOGIK (SIGMOID-INFERENZ) ---
def calculate_risk_inference(alter, last, thermik, kss_ausfall, integritaet):
    # Gewichtete Logik für die Risikoprognose
    z = (alter * 1.8) + (last * 3.2) + (thermik * 4.5) + (kss_ausfall * 5.0) + ((100 - integritaet) * 0.1)
    risk = 1 / (1 + np.exp(-(z - 7.0)))
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
    "1.4404 (Edelstahl)": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Prozess-Konfiguration")
    mat_name = st.selectbox("Werkstoff-Auswahl", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.01, 1.0, 0.18)
    d = st.number_input("Werkzeug-Durchmesser [mm]", 1.0, 50.0, 12.0)
    kss = st.toggle("Kühlschmierung (KSS)", value=True)
    
    st.divider()
    st.header("📡 Sensor-Parameter")
    sens_vibr = st.slider("Vibrations-Empfindlichkeit", 0.5, 5.0, 1.0)
    zyklus_sprung = st.number_input("Schrittweite [Zyklen]", 1, 100, 10)
    sim_takt = st.select_slider("Simulations-Takt (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. PHYSIK- & KI-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    
    # Drehmoment-Berechnung (Last)
    kc = m['kc1.1'] * (f ** -m['mc'])
    drehmoment = (kc * f * (d/2) * (d/2)) / 1000 # Vereinfachtes Moment Nm
    
    # Verschleiß & Thermik
    v_zuwachs = ((m['rate'] * (vc**1.5) * f) / (10000 if kss else 400)) * zyklus_sprung
    s['verschleiss'] += v_zuwachs
    t_ziel = 22 + (s['verschleiss'] * 1.5) + (vc * 0.2) + (0 if kss else 250)
    s['thermik'] += (t_ziel - s['thermik']) * 0.2
    
    # Vibration
    s['vibration'] = ((s['verschleiss'] * 0.1) + (vc * 0.01) + (100 - s['integritaet'])*0.2) * sens_vibr + s['seed'].normal(0,0.2)
    
    # KI-Inferenz
    s['risk'] = calculate_risk_inference(s['zyklus']/1000, drehmoment/50, s['thermik']/m['t_crit'], 1.0 if not kss else 0.0, s['integritaet'])
    
    # Aufschlüsselung Struktur-Verlust (XAI Deep Dive)
    loss_fatigue = (s['verschleiss'] / 100) * 0.05 * zyklus_sprung
    loss_load = (s['risk'] ** 3) * 1.2 * zyklus_sprung if s['risk'] > 0.2 else 0
    loss_thermal = (np.exp(max(0, s['thermik'] - m['t_crit']) / 50) - 1) * zyklus_sprung
    
    s['integritaet'] -= (loss_fatigue + loss_load + loss_thermal)
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0

    log = {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integritaet'], 'last': drehmoment, 'temp': s['thermik'], 'vib': s['vibration'], 'f_loss': loss_fatigue, 'l_loss': loss_load, 't_loss': loss_thermal}
    s['logs'].insert(0, log)
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration']})

# --- 6. UI DASHBOARD ---
tab1, tab2 = st.tabs(["📊 LIVE-MONITORING", "🧪 WAS-WÄRE-WENN ANALYSE"])

with tab1:
    if s['broken']: st.markdown('<div class="emergency-alert">🚨 SYSTEM-STOPP: STRUKTUR-INTEGRITÄT KRITISCH (WERKZEUGBRUCH)</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="glass-card">Struktur-Integrität<br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.2f}%</span></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="glass-card">KI-Bruchrisiko<br><span class="val-main">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="glass-card">Thermik<br><span class="val-main" style="color:#f85149">{s["thermik"]:.1f}°C</span></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="glass-card">Zerspanungs-Last<br><span class="val-main" style="color:#58a6ff">{s["logs"][0]["last"]:.1f} Nm</span></div>' if s['logs'] else '---', unsafe_allow_html=True)

    col_graph, col_xai = st.columns([2.5, 1.5])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Struktur-Integrität [%]", "Thermik [°C]", "Vibration [mm/s]", "KI-Bruchrisiko [%]"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#3fb950')), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#f85149')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#58a6ff')), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, line=dict(color='#e3b341')), 4, 1)
            fig.update_layout(height=700, template="plotly_dark", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_xai:
        st.markdown("### 🔍 XAI Deep-Monitor")
        x_html = ""
        for l in s['logs'][:10]:
            x_html += f"""
            <div style="border-left: 4px solid #e3b341; background: rgba(255,255,255,0.05); padding: 12px; margin-bottom: 10px; border-radius: 4px; font-size: 13px;">
                <b>Zyklus-Zeit: {l['zeit']} | Risiko: {l['risk']:.2%}</b><br>
                <span style="color:#f85149;">Integritäts-Verlust pro Schritt:</span><br>
                • Ermüdung: -{l['f_loss']:.5f}%<br>
                • Akut-Last: -{l['l_loss']:.5f}%<br>
                • Thermik-Gefüge: -{l['t_loss']:.5f}%
            </div>"""
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:680px; overflow-y:auto;">{x_html}</div>', height=700)

with tab2:
    st.header("🧪 Was-Wäre-Wenn Analyse")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        sim_alter = st.slider("Werkzeug-Alter [Zyklen]", 0, 2000, 200)
        sim_integ = st.slider("Vorschaden Integrität [%]", 0, 100, 100)
    with sc2:
        sim_last = st.slider("Last-Simulation [Nm]", 0, 150, 30)
        sim_temp = st.slider("Thermik-Simulation [°C]", 20, 900, 100)
        sim_kss = st.toggle("KSS-Ausfall simulieren", value=False)
    with sc3:
        r_sim = calculate_risk_inference(sim_alter/800, sim_last/50, sim_temp/500, 1.0 if sim_kss else 0.0, sim_integ)
        st.markdown(f'<div class="glass-card" style="text-align:center;">Prognostiziertes Risikon<h1 style="color:#e3b341; font-size:4rem;">{r_sim:.2%}</h1></div>', unsafe_allow_html=True)
        # Radar-Chart für Impact-Analyse
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[sim_alter/20, sim_last/1.5, sim_temp/9, (100 if sim_kss else 0)],
            theta=['Alter','Last','Thermik','KSS-Fehler'], fill='toself', line=dict(color='#e3b341')
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
b_start, b_reset = st.columns(2)
with b_start:
    if st.button("START / STOPP SIMULATION", use_container_width=True): s['active'] = not s['active']
with b_reset:
    if st.button("RESET (NEUES WERKZEUG)", use_container_width=True):
        st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42)}
        st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
