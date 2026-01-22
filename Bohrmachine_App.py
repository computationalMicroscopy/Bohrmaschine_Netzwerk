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
st.set_page_config(layout="wide", page_title="KI - Labor Bohrtechnik ULTIMATE", page_icon="⚙️")

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
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 800; margin: 5px 0; }
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
    score = (age_norm * 1.2) + (load_norm * 2.5) + (therm_norm * 4.0) + (cool_val * 3.0) + ((1.0 - health_norm) * 5.0)
    risk = 1 / (1 + np.exp(-(score - 3.5))) 
    return np.clip(risk, 0.01, 0.99)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 450},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 600},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SIDEBAR (ALLE REGLER ZURÜCK) ---
with st.sidebar:
    st.header("⚙️ Prozess-Parameter")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    st.divider()
    st.header("📡 Sensor-Konfiguration")
    sens_vib = st.slider("Vibrations-Empfindlichkeit", 0.1, 5.0, 1.0)
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)
    cycle_step = st.number_input("Schrittweite", 1, 50, 5)
    sim_speed = st.select_slider("Verzögerung (ms)", options=[500, 200, 100, 50, 0], value=50)

# --- 5. LOGIK (DETAILLIERTE ANALYSE) ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['cycle'] += cycle_step
    
    # Physikalische Berechnungen
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300)) * cycle_step
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2
    s['vib'] = ((s['wear'] * 0.05) + (vc * 0.01) + (100 - s['integrity']) * 0.2) * sens_vib + s['seed'].normal(0, 0.2)

    # KI-Risiko
    s['risk'] = get_continuous_risk(s['cycle']/800, (mc_raw*sens_load)/50, s['t_current']/mat['temp_crit'], 1.0 if not cooling else 0.0, s['integrity']/100)
    
    # Schadens-Splitting für XAI
    f_loss = (s['wear'] / 100) * 0.05 * cycle_step
    a_loss = (s['risk'] ** 2.5) * 0.7 * cycle_step if s['risk'] > 0.2 else 0
    t_loss = (np.exp((s['t_current'] - mat['temp_crit']) / 50) - 1) * cycle_step * 2 if s['t_current'] >= mat['temp_crit'] else 0
    
    s['integrity'] -= (f_loss + a_loss + t_loss)
    if s['integrity'] <= 0: s['broken'], s['active'], s['integrity'] = True, False, 0

    log_entry = {
        'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integrity'],
        'f_loss': f_loss, 'a_loss': a_loss, 't_loss': t_loss,
        'temp': s['t_current'], 'vib': s['vib'], 'mc': mc_raw
    }
    s['logs'].insert(0, log_entry)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'i': s['integrity'], 't': s['t_current'], 'v': s['vib']})

# --- 6. UI ---
tab1, tab2 = st.tabs(["📊 LIVE-MONITOR", "🧪 WAS-WÄRE-WENN ANALYSE"])

with tab1:
    if s['broken']: st.markdown('<div class="emergency-alert">🚨 TOTALAUSFALL</div>', unsafe_allow_html=True)
    
    col_met, col_gra, col_xai = st.columns([1, 2, 1.5])
    
    with col_met:
        st.markdown(f'<div class="glass-card">Integrität<br><span class="val-main" style="color:#3fb950">{s["integrity"]:.1f}%</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">KI-Risiko<br><span class="val-main" style="color:#e3b341">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">Temperatur<br><span class="val-main" style="color:#f85149">{s["t_current"]:.1f}°C</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">Vibration<br><span class="val-main" style="color:#58a6ff">{s["vib"]:.2f}</span></div>', unsafe_allow_html=True)

    with col_gra:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=df['c'], y=df['i'], name="Integrität", fill='tozeroy', line=dict(color='#3fb950')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['t'], name="Temp °C", line=dict(color='#f85149')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="Risiko %", line=dict(color='#e3b341')), row=3, col=1)
            fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with col_xai:
        st.markdown("### XAI Echtzeit-Analyse")
        xai_html = ""
        for l in s['logs'][:10]:
            xai_html += f"""
            <div style="border-bottom: 1px solid #333; padding: 10px; font-size: 13px;">
                <b style="color:#58a6ff;">[{l['zeit']}] Risiko: {l['risk']:.1%}</b><br>
                <span style="color:#f85149;">- Detaillierter Integritätsverlust:</span><br>
                • Abnutzung: -{l['f_loss']:.4f}% | • Überlast: -{l['a_loss']:.4f}% | • Hitze: -{l['t_loss']:.4f}%
            </div>"""
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:580px; overflow-y:auto;">{xai_html}</div>', height=600)

with tab2:
    st.header("🧪 Was-Wäre-Wenn Analyse")
    c1, c2 = st.columns([1, 2])
    with c1:
        s_age = st.slider("Alter", 0, 1000, 200)
        s_f = st.slider("Vorschub", 0.0, 1.0, 0.15)
        s_t = st.slider("Temp", 20, 800, 100)
        s_h = st.slider("Integrität", 0, 100, 100)
        s_c = st.toggle("Kühlung AUS", value=False)
    with c2:
        r = get_continuous_risk(s_age/800, s_f*5, s_t/500, 1.0 if s_c else 0.0, s_h/100)
        st.metric("Berechnetes Risiko", f"{r:.2%}")
        st.progress(r)
        if r > 0.7: st.error("Kritische Parameterkombination!")
        elif r > 0.3: st.warning("Erhöhter Verschleiß zu erwarten.")
        else: st.success("Sicherer Prozess.")

st.divider()
b1, b2 = st.columns(2)
with b1:
    if st.button("START / STOPP", use_container_width=True): s['active'] = not s['active']
with b2:
    if st.button("RESET", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if s['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
