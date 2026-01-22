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
st.set_page_config(layout="wide", page_title="KI - Digital Twin: Full XAI Drill Monitoring", page_icon="⚙️")

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
    .melt-warning {
        background: #f85149; color: white; padding: 10px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 15px;
        border: 2px solid #ffffff; animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.5; } }
    @keyframes pulse { 0% { box-shadow: 0 0 5px #f85149; } 50% { box-shadow: 0 0 25px #f85149; } 100% { box-shadow: 0 0 5px #f85149; } }
    .val-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.5rem; font-weight: 800; margin: 5px 0; }
    .ttf-val { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; }
    .terminal { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; height: 480px; background: #010409; padding: 15px; border-radius: 10px; border: 1px solid #30363d; color: #3fb950; overflow-y: auto; line-height: 1.6; }
    .xai-entry { margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 8px; }
    .xai-tag { color: #e3b341; font-weight: bold; font-size: 0.7rem; }
    .xai-val { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KI-ENGINE ---
@st.cache_resource
def get_engine():
    model = DiscreteBayesianNetwork([('Age', 'State'), ('Load', 'State'), ('Therm', 'State'), ('Cool', 'State')])
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.95], [0.05]])
    z_matrix = []
    for age in range(3):
        for load in range(2):
            for therm in range(2):
                for cool in range(2):
                    score = (age * 2) + (load * 4) + (therm * 7) + (cool * 8)
                    if score <= 3: v = [0.99, 0.005, 0.005]
                    elif score <= 8: v = [0.60, 0.35, 0.05]
                    elif score <= 12: v = [0.15, 0.45, 0.40]
                    elif score <= 16: v = [0.05, 0.15, 0.80]
                    else: v = [0.01, 0.04, 0.95]
                    z_matrix.append(v)
    cpd_state = TabularCPD('State', 3, np.array(z_matrix).T, ['Age', 'Load', 'Therm', 'Cool'], [3, 2, 2, 2])
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state)
    return VariableElimination(model)

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False,
        't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SIDEBAR ---
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
    cycle_step = st.number_input("Schrittweite", 1, 50, 1)
    sim_speed = st.select_slider("Verzögerung (ms)", options=[500, 200, 100, 50, 10, 0], value=50)
    sens_load = st.slider("Last-Empfindlichkeit", 0.1, 5.0, 1.0)

# --- 5. LOGIK ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += cycle_step
    
    # Physik & Verschleiß
    fc = mat['kc1.1'] * (f ** (1 - mat['mc'])) * (d / 2)
    mc_raw = (fc * d) / 2000
    s['wear'] += ((mat['wear_rate'] * (vc ** 1.8) * f) / (15000 if cooling else 300)) * cycle_step
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.2 + s['seed'].normal(0, 0.4)
    
    # KI Inferenz
    age_cat = 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2)
    load_cat = 1 if mc_raw > ((d * 2.2) / sens_load) else 0
    therm_cat = 1 if s['t_current'] > mat['temp_crit'] else 0
    cool_cat = 0 if cooling else 1
    engine = get_engine()
    s['risk'] = engine.query(['State'], evidence={'Age': age_cat, 'Load': load_cat, 'Therm': therm_cat, 'Cool': cool_cat}).values[2]

    # Integritätsverlust (Ermüdung + Risiko + Exponentielle Thermik)
    fatigue = (s['wear'] / 100) * 0.05 * cycle_step
    acute_damage = (s['risk'] ** 3) * 0.5 * cycle_step if s['risk'] > 0.4 else 0
    thermal_collapse = 0
    if s['t_current'] >= mat['temp_crit']:
        t_diff = s['t_current'] - mat['temp_crit']
        thermal_collapse = (np.exp(t_diff / 50) - 1) * cycle_step * 2
    
    s['integrity'] -= (fatigue + acute_damage + thermal_collapse)
    if s['integrity'] <= 0:
        s['broken'] = True
        s['active'] = False
        s['integrity'] = 0

    # Saubere Daten für das Logging (kein HTML hier!)
    zeit = time.strftime("%H:%M:%S")
    log_data = {
        'zeit': zeit, 'zyk': s['cycle'], 'risk': s['risk'], 'integ': s['integrity'],
        'age': ["NEU", "GEBR.", "ALT"][age_cat], 'load': "HOCH" if load_cat else "NORM",
        'therm': "CRIT" if s['t_current'] >= mat['temp_crit'] else "OK",
        'temp': s['t_current'], 'md': mc_raw, 'wear': s['wear'],
        'f_loss': fatigue, 'a_loss': acute_damage, 't_loss': thermal_collapse
    }
    s['logs'].insert(0, log_data)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'i': s['integrity']})

# --- 6. UI ---
st.title("KI - Digital Twin: Full XAI Drill Monitoring")

if st.session_state.twin['t_current'] >= mat['temp_crit'] and not st.session_state.twin['broken']:
    st.markdown(f'<div class="melt-warning">🔥 MATERIAL-KOLLAPS: TEMPERATUR ÜBER SCHMELZPUNKT ({mat["temp_crit"]}°C)!</div>', unsafe_allow_html=True)

if st.session_state.twin['broken']:
    st.error("🚨 KATASTROPHALER WERKZEUGAUSFALL: Strukturintegrität bei 0%!", icon="💥")

col_metrics, col_main, col_logs = st.columns([1, 2, 1.4])

with col_metrics:
    st.markdown(f'<div class="glass-card"><span class="val-title">Zyklus</span><br><span class="val-main" style="color:#58a6ff">{st.session_state.twin["cycle"]}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Struktur-Integrität</span><br><span class="val-main" style="color:#3fb950">{max(0, st.session_state.twin["integrity"]):.1f} %</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Bohrertemperatur</span><br><span class="val-main" style="color:#f85149">{st.session_state.twin["t_current"]:.1f} °C</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card"><span class="val-title">Verschleiß</span><br><span class="val-main" style="color:#e3b341">{st.session_state.twin["wear"]:.1f} %</span></div>', unsafe_allow_html=True)

with col_main:
    ttf = "---"
    if len(st.session_state.twin['history']) > 5:
        df_h = pd.DataFrame(st.session_state.twin['history'])
        z = np.polyfit(df_h['c'], df_h['w'], 1)
        ttf = max(0, int((100 - st.session_state.twin['wear']) / max(0.00001, z[0])))
    
    is_critical = st.session_state.twin['risk'] > 0.7 or st.session_state.twin['integrity'] < 40
    st.markdown(f'<div class="{"warning-card" if is_critical else "predictive-card"}"><span class="val-title">🔮 Predictive Maintenance TTF</span><br><div class="ttf-val">{ttf}</div><span class="val-title">Zyklen bis empfohlener Wartung</span></div>', unsafe_allow_html=True)
    
    if len(st.session_state.twin['history']) > 0:
        df_p = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['i'], name="Integrität %", fill='tozeroy', line=dict(color='#3fb950')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['t'], name="Temperatur °C", line=dict(color='#f85149')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r']*100, name="Risiko %", line=dict(color='#e3b341')), row=3, col=1)
        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_logs:
    st.markdown('<p class="val-title">Echtzeit XAI-System Monitor</p>', unsafe_allow_html=True)
    # Rendering erfolgt jetzt SICHER hier
    log_content = ""
    for l in st.session_state.twin['logs'][:15]:
        status_col = "#f85149" if l['risk'] > 0.6 else "#3fb950"
        log_content += f"""
        <div class="xai-entry">
            <b style="color:#58a6ff;">[{l['zeit']}] ZYK {l['zyk']}</b> | <b style="color:{status_col};">RISIKO: {l['risk']:.1%}</b><br>
            <span class="xai-tag">INTEGRITÄT:</span> {l['integ']:.2f}% | <span class="xai-tag">KI-EV:</span> {l['age']}/{l['load']}/{l['therm']}<br>
            <span class="xai-tag">SENSOR:</span> {l['temp']:.1f}°C / {l['md']:.1f}Nm / {l['wear']:.1f}%<br>
            <span class="xai-tag">LOSS:</span> F:{l['f_loss']:.3f} | L:{l['a_loss']:.3f} | <b style="color:#f85149;">T:{l['t_loss']:.3f}</b>
        </div>
        """
    st.markdown(f'<div class="terminal">{log_content}</div>', unsafe_allow_html=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("▶ START / STOPP", use_container_width=True, disabled=st.session_state.twin['broken']):
        st.session_state.twin['active'] = not st.session_state.twin['active']
with c2:
    if st.button("🔄 RESET & REPARATUR", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed / 1000)
    st.rerun()
