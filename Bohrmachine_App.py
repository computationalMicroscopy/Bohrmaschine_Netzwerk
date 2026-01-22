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
    .melt-warning {
        background: #f85149; color: white; padding: 10px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 15px;
        border: 2px solid #ffffff; animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.5; } }
    @keyframes pulse { 0% { box-shadow: 0 0 5px #f85149; } 50% { box-shadow: 0 0 25px #f85149; } 100% { box-shadow: 0 0 5px #f85149; } }
    .val-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .val-main { font-family: 'Inter', sans-serif; font-size: 2.2rem; font-weight: 800; margin: 5px 0; }
    .ttf-val { font-family: 'JetBrains Mono', monospace; font-size: 3.5rem; color: #e3b341; }
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
        't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750}
}

# --- 4. SEITENLEISTE ---
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
    cycle_step = st.number_input("Zyklus-Schrittweite", 1, 50, 1)
    sim_speed = st.select_slider("Verzögerung (ms)", options=[500, 200, 100, 50, 10, 0], value=50)

# --- 5. LOGIK ---
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
    engine = get_engine()
    s['risk'] = engine.query(['State'], evidence={'Age': age_cat, 'Load': load_cat, 'Therm': therm_cat, 'Cool': cool_cat}).values[2]

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

    zeit = time.strftime("%H:%M:%S")
    log_data = {
        'zeit': zeit, 'zyk': s['cycle'], 'risk': s['risk'], 'integ': s['integrity'],
        'age': ["NEUWERTIG", "GEBRAUCHT", "ALT"][age_cat], 
        'load_status': load_cat,
        'therm': "KRITISCH" if s['t_current'] >= mat['temp_crit'] else "STABIL",
        'temp': s['t_current'], 'md': mc_raw, 'wear': s['wear'], 'vib': s['vib'],
        'f_loss': fatigue, 'a_loss': acute_damage, 't_loss': thermal_collapse
    }
    s['logs'].insert(0, log_data)
    s['history'].append({'c': s['cycle'], 'r': s['risk'], 'w': s['wear'], 't': s['t_current'], 'i': s['integrity'], 'v': s['vib']})

# --- 6. BENUTZEROBERFLÄCHE ---
st.title("KI - Labor Bohrtechnik")

if st.session_state.twin['t_current'] >= mat['temp_crit'] and not st.session_state.twin['broken']:
    st.markdown(f'<div class="melt-warning">🔥 MATERIAL-KOLLAPS: TEMPERATUR ÜBER SCHMELZPUNKT ({mat["temp_crit"]}°C)!</div>', unsafe_allow_html=True)

if st.session_state.twin['broken']:
    st.error("🚨 KATASTROPHALER WERKZEUGAUSFALL: Strukturintegrität bei 0%!", icon="💥")

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
    
    is_critical = st.session_state.twin['risk'] > 0.7 or st.session_state.twin['integrity'] < 40
    st.markdown(f'<div class="{"warning-card" if is_critical else "predictive-card"}"><span class="val-title">🔮 Vorausschauende Wartung (TTF)</span><br><div class="ttf-val">{ttf}</div><span class="val-title">Zyklen bis empfohlener Wartung</span></div>', unsafe_allow_html=True)
    
    if len(st.session_state.twin['history']) > 0:
        df_p = pd.DataFrame(st.session_state.twin['history'])
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['i'], name="Integrität %", fill='tozeroy', line=dict(color='#3fb950')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['t'], name="Temp °C", line=dict(color='#f85149')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['v'], name="Vibration mm/s", line=dict(color='#58a6ff')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_p['c'], y=df_p['r']*100, name="Risiko %", line=dict(color='#e3b341')), row=4, col=1)
        fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

with col_protokoll:
    st.markdown('<p class="val-title">Echtzeit Analyse-Monitor (XAI)</p>', unsafe_allow_html=True)
    
    html_eintraege = ""
    for l in st.session_state.twin['logs'][:15]:
        status_farbe = "#f85149" if l['risk'] > 0.6 else "#3fb950"
        html_eintraege += f"""
        <div style="margin-bottom: 25px; border-bottom: 2px solid #333; padding-bottom: 15px; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #e1e4e8;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <b style="color:#58a6ff;">[{l['zeit']}] ZYKLUS: {l['zyk']}</b>
                <b style="color:{status_farbe};">KRITISCHES RISIKO: {l['risk']:.1%}</b>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.03); padding: 8px; border-radius: 4px; border-left: 3px solid #e3b341; margin-bottom: 8px;">
                <b style="color:#e3b341; font-size: 11px; text-transform: uppercase;">KI-ENTSCHEIDUNGSGRUNDLAGE (EVIDENZ):</b><br>
                <b>Werkzeugalter:</b> {l['age']} | <b>Lastzustand:</b> {"HOCH" if l['load_status'] else "NORMAL"} | <b>Thermischer Status:</b> {l['therm']}
            </div>

            <div style="background: rgba(248, 81, 73, 0.05); padding: 8px; border-radius: 4px; border-left: 3px solid #f85149; margin-bottom: 8px;">
                <b style="color:#f85149; font-size: 11px; text-transform: uppercase;">DETAILLIERTE URSACHENANALYSE DER MATERIALSCHÄDIGUNG:</b><br>
                <div style="margin-top:4px;">• <b>Stetiger Substanzverlust:</b> -{l['f_loss']:.4f}% <span style="color:#8b949e; font-size:11px;">(Verschleiß der Schneidkante)</span></div>
                <div>• <b>Mechanische Schadenslast:</b> -{l['a_loss']:.4f}% <span style="color:#8b949e; font-size:11px;">(Überbeanspruchung des Werkzeugkörpers)</span></div>
                <div>• <b>Thermische Gefügezerstörung:</b> <span style="color:#f85149;">-{l['t_loss']:.4f}%</span> <span style="color:#8b949e; font-size:11px;">(Hitzeinduzierter Härteverlust)</span></div>
            </div>

            <div style="color: #8b949e; font-size: 12px; padding: 4px;">
                <b>AKTUELL GEMESSEN:</b> {l['temp']:.1f}°C | Vibration: {l['vib']:.2f} mm/s | Drehmoment: {l['md']:.1f} Nm
            </div>
        </div>
        """
    
    voll_html = f"""
    <div style="background: #010409; padding: 15px; border-radius: 8px; border: 1px solid #30363d; height: 500px; overflow-y: auto; scrollbar-width: thin;">
        {html_eintraege}
    </div>
    """
    st.components.v1.html(voll_html, height=520)

st.divider()
c1, c2 = st.columns(2)
with c1:
    if st.button("▶ START / STOPP", use_container_width=True, disabled=st.session_state.twin['broken']):
        st.session_state.twin['active'] = not st.session_state.twin['active']
with c2:
    if st.button("🔄 ZWILLING REPARIEREN & RESET", use_container_width=True):
        st.session_state.twin = {'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 't_current': 22.0, 'vib': 0.1, 'seed': np.random.RandomState(42), 'risk': 0.0, 'integrity': 100.0}
        st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed / 1000)
    st.rerun()
