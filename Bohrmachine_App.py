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
st.set_page_config(layout="wide", page_title="KI - Labor Bohrtechnik", page_icon="⚙️")

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
    .warning-card {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.4) 0%, rgba(40, 0, 0, 0.9) 100%);
        border: 2px solid #f85149; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px;
        animation: pulse 1.5s infinite;
    }
    .emergency-alert {
        background: #f85149; color: white; padding: 20px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
        border: 4px solid #ffffff; animation: blinker 0.8s linear infinite;
        font-size: 1.5rem;
    }
    @keyframes blinker { 50% { opacity: 0.3; } }
    @keyframes pulse { 0% { box-shadow: 0 0 5px #f85149; } 50% { box-shadow: 0 0 40px #f85149; } 100% { box-shadow: 0 0 5px #f85149; } }
    .val-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 800; margin: 5px 0; }
    .ttf-val { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; }
    .melt-warning {
        background: #f85149; color: white; padding: 10px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 15px;
        border: 2px solid #ffffff; animation: blinker 1s linear infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-ENGINE ---
@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State'), ('Health', 'State')])
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

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SEITENLEISTE (PROZESSSTEUERUNG) ---
with st.sidebar:
    st.header("⚙️ Live-Prozess")
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
    cycle_step = st.number_input("Zyklus-Schrittweite", 1, 50, 1)
    sim_speed = st.select_slider("Verzögerung (ms)", options=[500, 200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK (LIVE) ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += cycle_step
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300)) * cycle_step
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.4)
    base_vib = (s['wear'] * 0.05) + (vc * 0.01)
    integrity_penalty = (100 - s['integrity']) * 0.2
    s['vib'] = (base_vib + integrity_penalty) * sens_vib + s['seed'].normal(0, 0.2)
    s['vib'] = max(0.1, s['vib'])

    age_cat = 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2)
    load_cat = 1 if mc_raw > ((d * 2.2) / sens_load) else 0
    therm_cat = 1 if s['t_current'] > mat['temp_crit'] else 0
    cool_cat = 0 if cooling else 1
    health_cat = 0 if s['integrity'] > 70 else (1 if s['integrity'] > 30 else 2)

    engine = get_engine()
    s['risk'] = engine.query(['State'], evidence={'Age': age_cat, 'Load': load_cat, 'Therm': therm_cat, 'Cool': cool_cat, 'Health': health_cat}).values[2]

    fatigue = (s['wear'] / 100) * 0.05 * cycle_step
    acute_damage = (s['risk'] ** 2.5) * 0.7 * cycle_step if s['risk'] > 0.2 else 0
    thermal_collapse = (np.exp((s['t_current'] - mat['temp_crit']) / 50) - 1) * cycle_step * 2 if s['t_current'] >= mat['temp_crit'] else 0
    
    s['integrity'] -= (fatigue + acute_damage + thermal_collapse)
    if s['integrity'] <= 0:
        s['broken'], s['active'], s['integrity'] = True, False, 0

    log_data = {'zeit': time.strftime("%H:%M:%S"), 'zyk': s['cycle'], 'risk': s['risk'], 'integ': s['integrity'], 'age': ["NEUWERTIG", "GEBRAUCHT", "ALT"][age_cat], 'health_str': ["STABIL", "GESCHWÄCHT", "KRITISCH"][health_cat], 'load_status': load_cat, 'therm': "KRITISCH" if s['t_current'] >= mat['temp_crit'] else "STABIL", 'temp': s['t_current'], 'md': mc_raw, 'wear': s['wear'], 'vib': s['vib'], 'f_loss': fatigue, 'a_loss': acute_damage, 't_loss': thermal_collapse}
    s['logs'].insert(0, log_data)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'i': s['integrity'], 'v': s['vib']})

# --- 6. UI ---
st.title("KI - Labor Bohrtechnik")

tab1, tab2 = st.tabs(["📊 LIVE-MONITORING", "🧪 WAS-WÄRE-WENN ANALYSE"])

with tab1:
    # (DEIN BISHERIGES UI - UNVERÄNDERT)
    if st.session_state.twin['broken']:
        st.markdown('<div class="emergency-alert">🚨 TOTALAUSFALL: WERKZEUG GEBROCHEN!</div>', unsafe_allow_html=True)
    elif st.session_state.twin['integrity'] < 15:
        st.markdown('<div class="emergency-alert">⚠️ SOFORT-STOPP: KRITISCHER VORSCHADEN (<15%)!</div>', unsafe_allow_html=True)
    elif st.session_state.twin['risk'] > 0.7:
        st.warning(f"⚠️ HOCHRISIKO-BEREICH: KI erkennt instabilen Systemzustand ({st.session_state.twin['risk']:.1%})!")

    col_metriken, col_haupt, col_protokoll = st.columns([1, 2, 1.4])
    with col_metriken:
        st.markdown(f'<div class="glass-card"><span class="val-title">Struktur-Integrität</span><br><span class="val-main" style="color:#3fb950">{max(0, st.session_state.twin["integrity"]):.1f} %</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><span class="val-title">Bohrertemperatur</span><br><span class="val-main" style="color:#f85149">{st.session_state.twin["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><span class="val-main" style="color:#58a6ff">{st.session_state.twin["vib"]:.2f} mm/s</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{st.session_state.twin["wear"]:.1f} %</span></div>', unsafe_allow_html=True)

    with col_haupt:
        ttf = "---"
        if len(st.session_state.twin['history']) > 5:
            df_h = pd.DataFrame(st.session_state.twin['history'])
            z = np.polyfit(df_h['c'], df_h['w'], 1)
            ttf = max(0, int((100 - st.session_state.twin['wear']) / max(0.00001, z[0])))
        st.markdown(f'<div class="{"warning-card" if st.session_state.twin["risk"] > 0.6 else "predictive-card"}"><span class="val-title">🔮 TTF Vorausschau</span><br><div class="ttf-val">{ttf}</div><span class="val-title">Zyklen bis Wartung</span></div>', unsafe_allow_html=True)
        if len(st.session_state.twin['history']) > 0:
            df_p = pd.DataFrame(st.session_state.twin['history'])
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['i'], name="Integrität %", fill='tozeroy', line=dict(color='#3fb950')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['t'], name="Temp °C", line=dict(color='#f85149')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['v'], name="Vibration mm/s", line=dict(color='#58a6ff')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r']*100, name="Risiko %", line=dict(color='#e3b341')), row=4, col=1)
            fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_protokoll:
        st.markdown('<p class="val-title">XAI Analyse-Monitor</p>', unsafe_allow_html=True)
        html_eintraege = "".join([f'<div style="margin-bottom:15px; border-bottom:1px solid #333; font-size:12px;"><b>[{l["zeit"]}] Risiko: {l["risk"]:.1%}</b><br>Basis: {l["age"]}, {l["health_str"]}, {l["therm"]}<br><span style="color:#f85149;">Schaden: -{(l["f_loss"]+l["a_loss"]+l["t_loss"]):.4f}%</span></div>' for l in st.session_state.twin['logs'][:10]])
        st.components.v1.html(f'<div style="background:#010409; color:white; padding:10px; border-radius:8px; height:480px; overflow-y:auto;">{html_eintraege}</div>', height=500)

with tab2:
    st.header("🧪 KI-Szenario Labor")
    st.info("Simuliere hier die Risiko-Einschätzung der KI für beliebige Parameter-Kombinationen, ohne den echten Bohrer zu belasten.")
    
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        s_age = st.radio("Werkzeug-Alter", ["Neuwertig", "Gebraucht", "Alt"])
        s_health = st.radio("Struktur-Integrität", ["Stabil (>70%)", "Geschwächt (30-70%)", "Kritisch (<30%)"])
        
    with c2:
        s_load = st.toggle("Hohe mechanische Last", value=False)
        s_therm = st.toggle("Thermische Überlast (>500°C)", value=False)
        s_cool = st.toggle("Kühlung AUS", value=False)
        
    # Mapping für KI
    evid = {
        'Age': ["Neuwertig", "Gebraucht", "Alt"].index(s_age),
        'Health': ["Stabil (>70%)", "Geschwächt (30-70%)", "Kritisch (<30%)"].index(s_health),
        'Load': 1 if s_load else 0,
        'Therm': 1 if s_therm else 0,
        'Cool': 1 if s_cool else 0
    }
    
    engine = get_engine()
    res = engine.query(['State'], evidence=evid).values
    
    with c3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Simuliertes KI-Risiko")
        sim_risk = res[2]
        
        fig_sim = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sim_risk * 100,
            title = {'text': "Bruch-Wahrscheinlichkeit (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#f85149" if sim_risk > 0.7 else "#e3b341"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(63, 185, 80, 0.2)"},
                    {'range': [30, 70], 'color': "rgba(227, 179, 65, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(248, 81, 73, 0.2)"}
                ]
            }
        ))
        fig_sim.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=300)
        st.plotly_chart(fig_sim, use_container_width=True)
        
        if sim_risk > 0.8:
            st.error("🚨 KRITISCH: Die KI würde in diesem Szenario den sofortigen Not-Halt einleiten!")
        elif sim_risk > 0.4:
            st.warning("⚠️ WARNUNG: Erhöhtes Risiko. Instandhaltung innerhalb der nächsten 5 Zyklen empfohlen.")
        else:
            st.success("✅ SICHER: Der Prozess läuft unter diesen Bedingungen stabil.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- GLOBAL CONTROLS ---
st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("▶ LIVE-PROZESS START / STOPP", use_container_width=True, disabled=st.session_state.twin['broken']):
        st.session_state.twin['active'] = not st.session_state.twin['active']
with c2:
    if st.button("🔄 ZWILLING RESET (NEUER BOHRER)", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed / 1000)
    st.rerun()
