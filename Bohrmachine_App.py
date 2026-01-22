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
st.set_page_config(layout="wide", page_title="KI - Labor Bohrtechnik Pro", page_icon="⚙️")

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
    .predictive-card {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.2) 0%, rgba(5, 7, 10, 0.8) 100%);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .emergency-alert {
        background: #f85149; color: white; padding: 20px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 4px solid #ffffff; animation: blinker 0.8s linear infinite;
        font-size: 1.5rem;
    }
    @keyframes blinker { 50% { opacity: 0.3; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-LOGIK (KONTINUIERLICH & XAI) ---
def get_continuous_risk(age_norm, load_norm, therm_norm, cool_val, health_norm):
    """Berechnet ein stufenloses Risiko (0.0 - 1.0) mittels Sigmoid-Logik"""
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
    "Baustahl (S235JR)": {"temp_crit": 450, "wear_mult": 0.005},
    "Edelstahl (1.4404)": {"temp_crit": 600, "wear_mult": 0.015}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Live-Parameter")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschwindigkeit vc", 20, 500, 160)
    f = st.slider("Vorschub f", 0.02, 1.0, 0.18)
    cooling = st.toggle("Kühlung aktiv", value=True)
    sim_speed = st.select_slider("Sim-Pause (ms)", options=[500, 200, 100, 50, 0], value=50)

# --- 5. LOGIK (LIVE-SIMULATION) ---
s = st.session_state.twin 

if s['active'] and not s['broken']:
    s['cycle'] += 5
    s['wear'] += (vc * f * mat['wear_mult']) if cooling else (vc * f * mat['wear_mult'] * 10)
    s['t_current'] = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['vib'] = (s['wear'] * 0.1) + (vc * 0.01) + s['seed'].normal(0, 0.3)
    
    s['risk'] = get_continuous_risk(
        age_norm = s['cycle']/800, 
        load_norm = (vc * f)/100, 
        therm_norm = s['t_current']/mat['temp_crit'], 
        cool_val = 1.0 if not cooling else 0.0, 
        health_norm = s['integrity']/100
    )
    
    s['integrity'] -= (0.1 + (s['risk'] * 0.8))
    if s['integrity'] <= 0:
        s['broken'], s['active'], s['integrity'] = True, False, 0

    reasons = []
    if s['t_current'] > mat['temp_crit']: reasons.append("Thermische Überlast")
    if s['risk'] > 0.6: reasons.append("KI warnt vor Strukturkollaps")
    if not cooling and vc > 200: reasons.append("Trockenlauf-Risiko")
    explanation = " & ".join(reasons) if reasons else "Stabiler Abtrag"

    log_entry = {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integrity'], 'expl': explanation}
    s['logs'].insert(0, log_entry)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'i': s['integrity']})

# --- 6. UI ---
tab1, tab2 = st.tabs(["📊 LIVE-MONITORING", "🧪 WAS-WÄRE-WENN (STUFENLOS)"])

with tab1:
    if s['broken']:
        st.markdown('<div class="emergency-alert">🚨 TOTALAUSFALL: BOHRER GEBROCHEN!</div>', unsafe_allow_html=True)
    
    c_met, c_graph, c_log = st.columns([1, 2, 1.2])
    
    with c_met:
        st.markdown(f'<div class="glass-card"><b>Integrität</b><br><h2>{s["integrity"]:.1f}%</h2></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><b>KI-Risiko</b><br><h2>{s["risk"]:.1%}</h2></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><b>Temperatur</b><br><h2>{s["t_current"]:.1f}°C</h2></div>', unsafe_allow_html=True)

    with c_graph:
        if len(s['history']) > 0:
            df = pd.DataFrame(s['history'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['c'], y=df['i'], name="Integrität %", line=dict(color='#3fb950', width=3)))
            fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="KI-Risiko %", line=dict(color='#e3b341', dash='dot')))
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Starte die Simulation, um Daten zu sehen.")

    with c_log:
        st.markdown("### XAI Analyse-Log")
        log_html = "".join([f'<div style="border-bottom:1px solid #333; padding:5px;"><small>{l["zeit"]}</small><br><b style="color:{"#f85149" if l["risk"] > 0.5 else "#3fb950"}">{l["expl"]}</b><br>Risiko: {l["risk"]:.1%}</div>' for l in s['logs'][:10]])
        st.components.v1.html(f'<div style="color:white; font-family:sans-serif; height:400px; overflow-y:auto;">{log_html}</div>', height=420)

with tab2:
    st.header("🧪 Interaktives KI-Szenario")
    cl1, cl2 = st.columns([1, 2])
    with cl1:
        sim_age = st.slider("Werkzeugalter (Zyklen)", 0, 1000, 200)
        sim_f = st.slider("Vorschub (mm/U)", 0.0, 1.0, 0.15)
        sim_temp = st.slider("Temperatur (°C)", 20, 800, 150)
        sim_health = st.slider("Struktur-Gesundheit (%)", 0, 100, 100)
        sim_cool = st.toggle("Kühlung deaktiviert", value=False)
    
    with cl2:
        res_risk = get_continuous_risk(sim_age/800, sim_f*5, sim_temp/500, 1.0 if sim_cool else 0.0, sim_health/100)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = res_risk * 100,
            title = {'text': "Bruch-Wahrscheinlichkeit (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if res_risk > 0.7 else "orange"},
                'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 75], 'color': "yellow"}, {'range': [75, 100], 'color': "red"}]
            }
        ))
        fig_gauge.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("#### KI-Begründung für dieses Szenario:")
        analysis = []
        if sim_health < 40: analysis.append("❌ **Vorschaden:** Die geringe Integrität ist der Haupttreiber des Risikos.")
        if sim_temp > 500: analysis.append("🔥 **Hitze:** Das Materialgefüge wird instabil.")
        if sim_cool and sim_temp > 300: analysis.append("❄️ **Kühlung:** Fehlende Kühlung bei Hitze wird kritisch bewertet.")
        if not analysis: analysis.append("✅ **Stabil:** Sicherer Betriebsbereich.")
        for a in analysis: st.write(a)

st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("▶ START / STOPP SIMULATION", use_container_width=True):
        st.session_state.twin['active'] = not st.session_state.twin['active']
with col2:
    if st.button("🔄 SYSTEM-RESET", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
