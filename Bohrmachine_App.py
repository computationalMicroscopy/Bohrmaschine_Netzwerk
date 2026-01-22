import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="KI-Expertensystem Bohrtechnik", page_icon="⚙️")

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
    .val-title { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; margin: 5px 0; }
    .emergency-alert {
        background: #f85149; color: white; padding: 20px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 4px solid #ffffff; animation: blinker 0.8s linear infinite;
        font-size: 1.5rem;
    }
    @keyframes blinker { 50% { opacity: 0.3; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-KERN (DYNAMISCHE INFERENZ) ---
def get_inference_risk(age_val, load_val, therm_val, cool_val, health_val):
    """ Berechnet ein präzises, stufenloses Risiko basierend auf gewichteter Inferenz """
    # Skalierung der Eingänge auf ein logistisches Modell (XAI-Simulation)
    z = (age_val * 1.5) + (load_val * 3.5) + (therm_val * 5.0) + (cool_val * 4.0) + ((100 - health_val) * 0.08)
    risk = 1 / (1 + np.exp(-(z - 6.5)))
    return np.clip(risk, 0.001, 0.999)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.25, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.45, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SIDEBAR (VOLLE KONTROLLE) ---
with st.sidebar:
    st.header("⚙️ Prozess-Steuerung")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 550, 180)
    f = st.slider("Vorschub f [mm/U]", 0.01, 1.2, 0.2)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 80.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    st.divider()
    st.header("📡 Sensorik & Takt")
    sens_vib = st.slider("Vibrations-Gain", 0.1, 10.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.1, 10.0, 1.0)
    cycle_step = st.number_input("Schrittweite [Zyklen]", 1, 200, 10)
    sim_speed = st.select_slider("Sim-Speed (ms)", options=[500, 200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK (VOLLE PHYSIK-ENGINE) ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['cycle'] += cycle_step
    
    # 1. Kraft & Drehmoment
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000 # Nm
    
    # 2. Verschleiß & Temperatur
    wear_inc = ((mat['wear_rate'] * (vc ** 1.9) * f) / (15000 if cooling else 200)) * cycle_step
    s['wear'] += wear_inc
    target_t = 22 + (s['wear'] * 1.6) + (vc * 0.2) + (0 if cooling else 280)
    s['t_current'] += (target_t - s['t_current']) * 0.25 + s['seed'].normal(0, 0.4)
    
    # 3. Vibrationen
    s['vib'] = ((s['wear'] * 0.08) + (vc * 0.012) + (100 - s['integrity']) * 0.3) * sens_vib + s['seed'].normal(0, 0.3)
    
    # 4. KI-Inferenz
    s['risk'] = get_inference_risk(s['cycle']/800, (mc_raw*sens_load)/50, s['t_current']/mat['temp_crit'], 1.0 if not cooling else 0.0, s['integrity'])
    
    # 5. Struktur-Schaden (Detailliertes XAI-Modell)
    f_loss = (s['wear'] / 100) * 0.03 * cycle_step
    a_loss = (s['risk'] ** 2.8) * 0.9 * cycle_step if s['risk'] > 0.15 else 0
    t_loss = (np.exp(max(0, s['t_current'] - mat['temp_crit']) / 45) - 1) * cycle_step * 3
    
    s['integrity'] -= (f_loss + a_loss + t_loss)
    if s['integrity'] <= 0: s['broken'], s['active'], s['integrity'] = True, False, 0

    log_entry = {
        'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integrity'],
        'f_loss': f_loss, 'a_loss': a_loss, 't_loss': t_loss,
        'temp': s['t_current'], 'vib': s['vib'], 'mc': mc_raw, 'wear': s['wear']
    }
    s['logs'].insert(0, log_entry)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'i': s['integrity'], 't': s['t_current'], 'v': s['vib'], 'w': s['wear']})

# --- 6. UI DASHBOARD ---
tab1, tab2 = st.tabs(["📊 LIVE-PROZESS-MONITORING", "🧪 VOLLSTÄNDIGE WAS-WÄRE-WENN ANALYSE"])

with tab1:
    if s['broken']: st.markdown('<div class="emergency-alert">🚨 KRITISCHER FEHLER: WERKZEUG-BRUCH DETEKTIERT!</div>', unsafe_allow_html=True)
    
    # Metriken (Großanzeige)
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="glass-card"><span class="val-title">Struktur-Integrität</span><br><p class="val-main" style="color:#3fb950">{s["integrity"]:.2f}%</p></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="glass-card"><span class="val-title">KI-Bruchrisiko</span><br><p class="val-main" style="color:#e3b341">{s["risk"]:.2%}</p></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><p class="val-main" style="color:#f85149">{s["t_current"]:.1f}°C</p></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="glass-card"><span class="val-title">Last (M_d)</span><br><p class="val-main" style="color:#58a6ff">{s["logs"][0]["mc"]:.2f} Nm</p></div>' if s['logs'] else '---', unsafe_allow_html=True)

    col_g, col_x = st.columns([2.3, 1.7])
    with col_g:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Struktur-Gesundheit (%)", "Thermik (°C)", "Vibration (mm/s)", "KI-Risiko (%)"))
            fig.add_trace(go.Scatter(x=df['c'], y=df['i'], name="Integrität", fill='tozeroy', line=dict(color='#3fb950', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['t'], name="Temp", line=dict(color='#f85149')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['v'], name="Vib", line=dict(color='#58a6ff')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="Risiko", line=dict(color='#e3b341')), row=4, col=1)
            fig.update_layout(height=750, template="plotly_dark", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_x:
        st.markdown("### 🔍 XAI-Matrix (Ursachen-Analyse)")
        x_html = ""
        for l in s['logs'][:15]:
            status_color = "#f85149" if l['risk'] > 0.6 else ("#e3b341" if l['risk'] > 0.2 else "#3fb950")
            x_html += f"""
            <div style="border-left: 5px solid {status_color}; background: rgba(255,255,255,0.04); padding: 15px; margin-bottom: 12px; border-radius: 5px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; opacity:0.8;">
                    <b>ZYKLUS: {l['zeit']}</b>
                    <b style="color:{status_color};">PROGNOSE: {l['risk']:.2%}</b>
                </div>
                <div style="margin-top:10px; font-size:13px;">
                    <b>Sensorwerte:</b> {l['temp']:.1f}°C | {l['vib']:.2f} mm/s | {l['mc']:.2f} Nm<br>
                    <hr style="border:0; border-top:1px solid #444; margin:8px 0;">
                    <span style="color:#f85149; font-weight:bold;">Integritäts-Verlust Aufschlüsselung:</span><br>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; font-family: 'JetBrains Mono'; font-size:11px; margin-top:5px;">
                        <span>Ermüdung:</span> <span>-{l['f_loss']:.5f}%</span>
                        <span>Akutlast:</span> <span>-{l['a_loss']:.5f}%</span>
                        <span>Thermik:</span> <span>-{l['t_loss']:.5f}%</span>
                        <span style="font-weight:bold;">SUMME:</span> <span style="font-weight:bold;">-{(l['f_loss']+l['a_loss']+l['t_loss']):.5f}%</span>
                    </div>
                </div>
            </div>"""
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:730px; overflow-y:auto; padding-right:10px;">{x_html}</div>', height=750)

with tab2:
    st.header("🧪 Was-Wäre-Wenn Simulations-Labor")
    st.markdown("Testen Sie hier extreme Parameterkombinationen, um das Verhalten der KI-Inferenz zu verstehen.")
    
    c1, c2, c3 = st.columns([1.2, 1.2, 2.5])
    
    with c1:
        st.subheader("🔧 Mechanik & Alter")
        sim_age = st.slider("Werkzeugalter [Zyklen]", 0, 2000, 500)
        sim_f = st.slider("Vorschub-Last [mm/U]", 0.0, 1.5, 0.2)
        sim_health = st.slider("Vorschaden (Integrität %)", 0, 100, 100)
    
    with c2:
        st.subheader("🔥 Thermik & Medium")
        sim_temp = st.slider("Temperatur-Sensor [°C]", 20, 1000, 150)
        sim_cool = st.checkbox("Kühlschmierung AUSGEFALLEN", value=False)
        sim_load_gain = st.slider("Zusatz-Vibrationen", 0.0, 10.0, 1.0)
    
    with c3:
        # Simulations-Berechnung
        sim_risk = get_inference_risk(sim_age/800, sim_f*4, sim_temp/500, 1.0 if sim_cool else 0.0, sim_health)
        
        st.markdown(f'<div class="glass-card" style="text-align:center;">'
                    f'<span class="val-title">Simulierte Bruchwahrscheinlichkeit</span><br>'
                    f'<h1 style="font-size:5rem; color:{"#f85149" if sim_risk > 0.7 else "#e3b341"}; margin:20px 0;">{sim_risk:.2%}</h1>'
                    f'</div>', unsafe_allow_html=True)
        
        # Expertensystem Begründung
        st.markdown("### 🤖 KI-Logik-Begründung")
        reasons = []
        if sim_health < 30: reasons.append("❌ **Kritische Substanz:** Die verbleibende Integrität ist zu gering für mechanische Lasten.")
        if sim_temp > 600: reasons.append("🔥 **Thermischer Stress:** Gefügeveränderungen durch Hitze dominieren das Risiko.")
        if sim_age > 1200: reasons.append("⏳ **Degradation:** Das Werkzeug hat sein statistisches Lebensende erreicht.")
        if sim_cool and sim_temp > 400: reasons.append("❄️ **Kühlung fehlt:** Ohne KSS ist kein stabiler Prozess möglich.")
        if not reasons: reasons.append("✅ **Sicherer Bereich:** Alle Parameter sind innerhalb der Toleranzgrenzen.")
        
        for r in reasons:
            st.write(r)

# --- GLOBAL CONTROLS ---
st.divider()
cl1, cl2 = st.columns(2)
with cl1:
    if st.button("▶ SIMULATION START / STOPP", use_container_width=True): s['active'] = not s['active']
with cl2:
    if st.button("🔄 VOLLSTÄNDIGER RESET (NEUES WERKZEUG)", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if s['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
