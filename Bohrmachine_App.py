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
    .xai-critical { color: #f85149; font-weight: bold; }
    .xai-info { color: #58a6ff; }
    .predictive-card {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.2) 0%, rgba(5, 7, 10, 0.8) 100%);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px;
    }
    .ttf-val { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; }
    .emergency-alert {
        background: #f85149; color: white; padding: 20px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 4px solid #ffffff; animation: blinker 0.8s linear infinite;
        font-size: 1.5rem;
    }
    @keyframes blinker { 50% { opacity: 0.3; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERBESSERTE KI-LOGIK (GRAU-STUFEN) ---
@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'), ('Health', 'State')])
    # CPDs bleiben für die Inferenz-Struktur gleich, aber wir nutzen Interpolation für die Anzeige
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.95], [0.05]])
    cpd_health = TabularCPD('Health', 3, [[0.33], [0.33], [0.34]])
    z_matrix = []
    for age in range(3):
        for load in range(2):
            for therm in range(2):
                for cool in range(2):
                    for health in range(3):
                        score = (age * 1.5) + (load * 3) + (therm * 5) + (cool * 6) + (health * 8)
                        if score <= 4: v = [0.99, 0.005, 0.005]
                        elif score <= 10: v = [0.60, 0.35, 0.05]
                        elif score <= 16: v = [0.15, 0.45, 0.40]
                        else: v = [0.01, 0.04, 0.95]
                        z_matrix.append(v)
    cpd_state = TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool', 'Health'], [3, 2, 2, 2, 3])
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_health, cpd_state)
    return VariableElimination(model)

def get_continuous_risk(age_val, load_val, therm_val, cool_val, health_val):
    """Berechnet ein fließendes Risiko statt starrer Kategorien"""
    base_score = (age_val * 0.15) + (load_val * 0.3) + (therm_val * 0.4) + (cool_val * 0.5) + ((100-health_val) * 0.01)
    risk = 1 / (1 + np.exp(- (base_score - 1.5) * 4)) # Sigmoid-Funktion für weiche Übergänge
    return np.clip(risk, 0.01, 0.99)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 450},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 600}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Live-Parameter")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    cooling = st.toggle("Kühlschmierung aktiv", value=True)
    sim_speed = st.select_slider("Speed (ms)", options=[200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += 5
    
    # Physikalische Simulation
    s['wear'] += (vc * f * 0.005) if cooling else (vc * f * 0.05)
    s['t_current'] = 22 + (s['wear'] * 1.2) + (vc * 0.3) + (0 if cooling else 300)
    s['vib'] = (s['wear'] * 0.08) + (vc * 0.01) + s['seed'].normal(0, 0.5)
    
    # KI Risiko (Kontinuierlich)
    s['risk'] = get_continuous_risk(s['cycle']/500, f*5, s['t_current']/mat['temp_crit'], 1 if not cooling else 0, s['integrity'])
    
    # Integritätsverlust
    s['integrity'] -= (0.05 + (s['risk'] * 0.5))
    if s['integrity'] <= 0: s['broken'], s['active'], s['integrity'] = True, False, 0

    # XAI Text-Generierung
    reasons = []
    if s['t_current'] > mat['temp_crit']: reasons.append(f"Kritische Hitze ({s['t_current']:.0f}°C)")
    if s['vib'] > 5: reasons.append("Hohe Vibration detektiert")
    if s['integrity'] < 40: reasons.append("Strukturelle Schwächung")
    explanation = " & ".join(reasons) if reasons else "Normaler Verschleißprozess"

    log_data = {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integrity'], 'expl': explanation, 'temp': s['t_current']}
    s['logs'].insert(0, log_data)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'i': s['integrity']})

# --- 6. UI ---
tab1, tab2 = st.tabs(["📊 LIVE-SYSTEM", "🧪 WAS-WÄRE-WENN (SMOOTH)"])

with tab1:
    if st.session_state.twin['broken']: st.markdown('<div class="emergency-alert">🚨 TOTALAUSFALL</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown(f'<div class="predictive-card">Integrität: {st.session_state.twin["integrity"]:.1f}% | Risiko: {st.session_state.twin["risk"]:.1%}</div>', unsafe_allow_html=True)
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['c'], y=df['i'], name="Integrität", line=dict(color='#3fb950')))
            fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="KI-Risiko %", line=dict(color='#e3b341')))
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("XAI Logbuch")
        for l in st.session_state.twin['logs'][:8]:
            color = "#f85149" if l['risk'] > 0.5 else "#3fb950"
            st.markdown(f"""<div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                <small>{l['zeit']}</small><br>
                <b>Status: {l['expl']}</b><br>
                <span style="color:{color}">Risiko: {l['risk']:.1%}</span>
            </div>""", unsafe_allow_html=True)

with tab2:
    st.subheader("Stufenlose Szenario-Analyse")
    st.write("Verstelle die Regler und beobachte, wie sich die Wahrscheinlichkeit im Prozentbereich verändert.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        sim_age = st.slider("Werkzeug-Alter (Zyklen)", 0, 1000, 100)
        sim_f = st.slider("Vorschub (mm/U)", 0.0, 1.0, 0.2)
        sim_temp = st.slider("Temperatur (°C)", 20, 800, 100)
        sim_cool = st.checkbox("Kühlung inaktiv", value=False)
        sim_health = st.slider("Aktuelle Integrität (%)", 0, 100, 100)
    
    with c2:
        res_risk = get_continuous_risk(sim_age/500, sim_f*5, sim_temp/500, 1 if sim_cool else 0, sim_health)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = res_risk * 100,
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}}
        ))
        fig_gauge.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Dynamische Analyse des Szenarios
        st.markdown("### KI-Urteilsbegründung für dieses Szenario:")
        if sim_temp > 500: st.write("- 🚩 **Gefahr:** Die Temperatur ist für das Materialgefüge kritisch.")
        if sim_age > 700: st.write("- 🚩 **Gefahr:** Das hohe Alter führt zu mikroskopischen Rissen.")
        if sim_cool and sim_temp > 300: st.write("- 🚩 **Gefahr:** Fehlende Kühlung beschleunigt den Kollaps.")
        if res_risk < 0.2: st.write("- ✅ **Sicher:** Alle Parameter liegen im grünen Bereich.")

st.divider()
if st.button("START / STOPP", use_container_width=True):
    st.session_state.twin['active'] = not st.session_state.twin['active']
if st.button("RESET"):
    st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000)
    st.rerun()
