import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & INDUSTRIAL DESIGN ---
st.set_page_config(layout="wide", page_title="TwinPro V5.5 | Professional AI Lab", page_icon="⚙️")

if 'twin' not in st.session_state:
    st.session_state.twin = {
        'zyklus': 0.0, 'zyklen_anzahl': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }

s = st.session_state.twin

MATERIALIEN = {
    "Baustahl (1.0037)": {"kc1.1": 1800, "mc": 0.25, "wear_factor": 0.004, "t_crit": 450, "color": "#4a5568"}, 
    "Vergütungsstahl (1.7225)": {"kc1.1": 2100, "mc": 0.24, "wear_factor": 0.012, "t_crit": 550, "color": "#2d3748"}, 
    "Titanlegierung (3.7165)": {"kc1.1": 2500, "mc": 0.23, "wear_factor": 0.055, "t_crit": 650, "color": "#718096"},
    "Edelstahl (1.4301)": {"kc1.1": 2300, "mc": 0.22, "wear_factor": 0.038, "t_crit": 600, "color": "#a0aec0"}
}

# --- SIDEBAR: LIVE-MASCHINEN-STEUERUNG ---
with st.sidebar:
    st.header("🎮 Live-Maschinen-Steuerung")
    mat_name = st.selectbox("Eingespanntes Material", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("Live vc [m/min]", 30, 350, 100)
    f = st.slider("Live Vorschub f [mm/U]", 0.05, 0.60, 0.15)
    d = st.slider("Bohrer-Durchmesser d [mm]", 5.0, 32.0, 12.0)
    kuehlung = st.toggle("KSS Kühlung Aktiv", value=True)
    
    st.divider()
    st.header("🎛️ System-Performance")
    schrittweite = st.number_input("Zeitskalierungsfaktor", min_value=1, max_value=1000, value=5)
    taktzeit = st.select_slider("Taktung (ms)", options=[500, 200, 100, 0], value=100)
    
    sensor_temp_gain = st.slider("Sensor-Empfindlichkeit (T)", 0.5, 2.5, 1.0)
    sensor_vibr_gain = st.slider("Sensor-Empfindlichkeit (V)", 0.5, 2.5, 1.0)

# DYNAMISCHE ANIMATIONS-ZEIT
basis_zyklus_zeit = 2.5 
dynamische_animations_dauer = max(0.05, basis_zyklus_zeit / schrittweite)

# --- GLOBAL DYNAMIC CSS ---
st.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
    .stApp {{ background-color: #06090e; color: #c9d1d9; font-family: 'Inter', sans-serif; }}
    .main-title {{ font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 20px; background: linear-gradient(90deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .glass-card {{ background: rgba(16, 22, 30, 0.9); border: 1px solid #30363d; border-radius: 12px; padding: 15px; margin-bottom: 10px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
    .val-title {{ font-size: 0.9rem; color: #8b949e; text-transform: uppercase; font-weight: 700; }}
    .val-main {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; color: #f0f6fc; display: block; }}
    .xai-container {{ height: 500px; overflow-y: auto; padding-right: 10px; border-radius: 8px; background: #0d1117; padding: 10px; border: 1px solid #30363d; }}
    .xai-card {{ background: #161b22; border-left: 5px solid #58a6ff; padding: 15px; border-radius: 6px; margin-bottom: 10px; }}
    .reason-text {{ color: #f0f6fc; font-size: 1.1rem; font-weight: 700; margin-bottom: 5px; }}
    .action-text {{ color: #ff7b72; font-weight: bold; font-size: 1rem; border-top: 1px solid #30363d; padding-top: 5px; margin-top: 5px; }}
    .diag-badge {{ background: #23426f; color: #58a6ff; border: 1px solid #388bfd; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 800; }}
    
    @keyframes tool_feed_optimized {{
        0%, 15%, 90%, 100% {{ transform: translateY(-35px); }}      
        25% {{ transform: translateY(-2px); }}      
        65%, 75% {{ transform: translateY(22px); }}      
    }}
    @keyframes vfx_sync_gate {{
        0%, 24%, 76%, 100% {{ opacity: 0; }}
        25%, 75% {{ opacity: 1; }}
    }}
    @keyframes led_pulse {{ 0%, 100% {{ box-shadow: 0 0 8px var(--led-color); }} 50% {{ box-shadow: 0 0 20px var(--led-color); }} }}
    </style>
""")

st.html('<div class="main-title">🚀 TwinPro | Industrie-Zwillings-Dashboard</div>')

# --- DIAGNOSE ENGINE ---
def compute_sensor_diagnostics(vals, settings, integrity, kuehlung, m):
    M, T, V, F = vals['M'], vals['T'], vals['V'], vals['F']
    vc, f, d = settings['vc'], settings['f'], settings['d']
    crit_torque, crit_force = 0.12 * (d**3), 320 * (d**2)
    t_crit = m['t_crit']
    
    evidenz = {
        "Mechanische Torsions-Überlast": min(100.0, (M/crit_torque)*100) if M > crit_torque*0.8 else 0,
        "Thermische Gefüge-Erweichung": min(100.0, (T/t_crit)*100) if T > t_crit*0.8 else 0,
        "Regeneratives Rattern": min(100.0, (V/8.0)*100) if V > 5.0 else 0,
        "Axiale Schaft-Knickung": min(100.0, (F/crit_force)*100) if F > crit_force*0.8 else 0,
        "Normalbetrieb": 10.0
    }
    top_reason = max(evidenz, key=evidenz.get)
    mapping = {
        "Mechanische Torsions-Überlast": {"diag": "CRITICAL TORQUE", "exp": "Das Drehmoment übersteigt die Scherspannungsgrenze.", "act": "Vorschub f senken!"},
        "Thermische Gefüge-Erweichung": {"diag": "THERMAL OVERLOAD", "exp": "Überhitzung führt zu Härteverlust an der Schneide.", "act": "Schnittgeschwindigkeit vc senken!"},
        "Regeneratives Rattern": {"diag": "RESONANT CHATTER", "exp": "Vibrationen stören die Spanbildung.", "act": "vc um 15% ändern!"},
        "Axiale Schaft-Knickung": {"diag": "AXIAL BUCKLED", "exp": "Knickgefahr des Schafts durch zu hohe Axialkraft.", "act": "f sofort reduzieren!"},
        "Normalbetrieb": {"diag": "NOMINAL", "exp": "Prozess im grünen Bereich.", "act": "Keine Korrektur nötig."}
    }
    res = mapping.get(top_reason)
    res["snapshot"] = f"M: {M:.1f}Nm | T: {T:.0f}°C | V: {V:.1f}mm/s"
    return res, sorted(evidenz.items(), key=lambda x: x[1], reverse=True)

# --- PHYSIK-ENGINE ---
n = (vc * 1000) / (np.pi * d) if d > 0 else 0
if s['active'] and not s['broken']:
    dt = (taktzeit / 1000.0 if taktzeit > 0 else 0.05) * schrittweite
    alt_z = int(s['zyklus'] / 2.5)
    s['zyklus'] += dt
    neu_z = int(s['zyklus'] / 2.5)
    if neu_z > alt_z: s['zyklen_anzahl'] += (neu_z - alt_z)
    
    kc = m['kc1.1'] * ((f/2.0) ** -m['mc'])
    s['drehmoment'] = (f * (d**2) * kc) / 8000.0
    s['vorschubkraft'] = (0.5 * d * f * kc) * 1.3
    t_target = 22.0 + ((s['drehmoment'] * n * 0.00013) / (4.5 if kuehlung else 0.8)) * sensor_temp_gain
    s['thermik'] += (t_target - s['thermik']) * 0.15
    s['vibration'] = (0.2 + (s['drehmoment'] * 0.08)) * sensor_vibr_gain
    
    wear = (m['wear_factor'] * ((vc/120)**1.5) * f * np.exp(max(0, s['thermik']-m['t_crit'])/30)) * dt * 5
    s['integritaet'] = max(0.0, s['integritaet'] - wear)
    
    diag, ev = compute_sensor_diagnostics({'M':s['drehmoment'], 'T':s['thermik'], 'V':s['vibration'], 'F':s['vorschubkraft']}, {'vc':vc, 'f':f, 'd':d}, s['integritaet'], kuehlung, m)
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'info': diag, 'evidenz': ev})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'm': s['drehmoment'], 't': s['thermik'], 'v': s['vibration'], 'f': s['vorschubkraft']})
    if s['integritaet'] <= 0 or s['drehmoment'] > (0.12 * d**3)*1.1: s['broken'] = True

# --- UI: ANIMATION & KPI ---
col_vfx, col_kpi = st.columns([1.3, 4])
with col_vfx:
    # Skalierter Bohrer
    scale = d / 12.0
    dw, dtp = int(44 * scale), int(21 * scale)
    led_c = "#ff4400" if s['risk'] > 0.7 else ("#00b4ff" if kuehlung else "#2ea44f")
    if s['broken']:
        drill_render = f'<div style="width:{dw}px; height:60px; background:#333; transform:rotate(-30deg); border-bottom:2px solid red;"></div>'
    else:
        drill_render = f"""
        <div style="animation: tool_feed_optimized {dynamische_animations_dauer}s infinite ease-in-out;">
            <div style="width:{dw}px; height:110px; background:repeating-linear-gradient(135deg, #111, #444 15px, #888 20px, #444 25px); box-shadow: inset 4px 0 10px #000;"></div>
            <div style="background:rgb({min(255, int(s['thermik']))}, 50, 50); width:{dw}px; height:{dtp}px; clip-path:polygon(0% 0%, 100% 0%, 50% 100%);"></div>
        </div>"""
    
    st.html(f"""
        <div class="glass-card" style="height:350px; display:flex; flex-direction:column; justify-content:center; align-items:center; overflow:hidden;">
            <div style="--led-color:{led_c}; width:60px; height:10px; background:{led_c}; border-radius:5px; animation: led_pulse 1s infinite; margin-bottom:20px;"></div>
            {drill_render}
            <div style="width:120%; height:20px; background:linear-gradient(to bottom, {m['color']}, #000); margin-top:-2px;"></div>
            <div style="margin-top:10px; font-weight:800; color:{led_c};">STATUS: {"CRASH" if s['broken'] else "LIVE"}</div>
        </div>
    """)

with col_kpi:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.html(f'<div class="glass-card"><span class="val-title">Zyklen</span><span class="val-main" style="color:#bc8cff">{s["zyklen_anzahl"]}</span></div>')
    k2.html(f'<div class="glass-card"><span class="val-title">Schnittzeit</span><span class="val-main">{s["zyklus"]:.1f}s</span></div>')
    k3.html(f'<div class="glass-card"><span class="val-title">Integrität</span><span class="val-main" style="color:#2ea44f">{s["integritaet"]:.1f}%</span></div>')
    k4.html(f'<div class="glass-card"><span class="val-title">Drehmoment</span><span class="val-main" style="color:#bc8cff">{s["drehmoment"]:.1f}Nm</span></div>')
    k5.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><span class="val-main" style="color:#ff7b72">{s["thermik"]:.0f}°C</span></div>')
    k6.html(f'<div class="glass-card"><span class="val-title">Schwingung</span><span class="val-main">{s["vibration"]:.2f}</span></div>')

# --- TABS: LIVE VS LABOR ---
t1, t2 = st.tabs(["📈 Echtzeit-Telemetrie", "🔬 Was-Wäre-Wenn Simulations-Labor"])

with t1:
    g_col, l_col = st.columns([2, 1])
    with g_col:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', name="Integrität", line=dict(color='#2ea44f')), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['m'], name="Nm", line=dict(color='#bc8cff')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], name="°C", line=dict(color='#ff7b72')), 2, 1)
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with l_col:
        st.markdown("**🤖 KI-Diagnostic Feed**")
        if s['logs']:
            st.html('<div class="xai-container">' + "".join([f'<div class="xai-card"><span class="diag-badge">{lg["info"]["diag"]}</span><div class="reason-text">{lg["info"]["exp"]}</div><div class="action-text">Lösung: {lg["info"]["act"]}</div></div>' for lg in s['logs'][:5]]) + '</div>')

with t2:
    st.markdown("### 🧪 Labor-Sandbox: Szenario-Analyse")
    st.info("Hier kannst du Parameter testen, ohne die Live-Maschine zu beeinflussen. Die Vorhersage berechnet die theoretische Lebensdauer bei diesen Werten.")
    
    lab_col1, lab_col2 = st.columns([1, 1])
    with lab_col1:
        st.subheader("Hypothetische Parameter")
        l_mat = st.selectbox("Labor-Werkstoff", list(MATERIALIEN.keys()), key="lab_m")
        l_vc = st.slider("Labor vc [m/min]", 30, 350, 100, key="lab_vc")
        l_f = st.slider("Labor f [mm/U]", 0.05, 0.60, 0.15, key="lab_f")
        l_d = st.slider("Labor d [mm]", 5.0, 32.0, 12.0, key="lab_d")
        l_k = st.toggle("Labor KSS", value=True, key="lab_k")
    
    with lab_col2:
        st.subheader("Prädiktive FEM-Ergebnisse")
        lm = MATERIALIEN[l_mat]
        ln = (l_vc * 1000) / (np.pi * l_d)
        lkc = lm['kc1.1'] * ((l_f/2.0) ** -lm['mc'])
        lm_est = (l_f * (l_d**2) * lkc) / 8000.0
        lt_est = 22.0 + ((lm_est * ln * 0.00013) / (4.5 if l_k else 0.8))
        
        # Standzeit-Vorhersage
        l_therm_accel = np.exp(max(0, lt_est-lm['t_crit'])/30)
        l_wear_cycle = (lm['wear_factor'] * ((l_vc/120)**1.5) * l_f * l_therm_accel) * 2.5 * 5
        pred_cycles = int(100 / max(0.001, l_wear_cycle))
        
        lc1, lc2 = st.columns(2)
        lc1.metric("Erwartetes Moment", f"{lm_est:.1f} Nm")
        lc1.metric("Erwartete Temp.", f"{lt_est:.0f} °C")
        lc2.metric("Sicherheit (Torsion)", f"{max(0, 100-(lm_est/(0.12*l_d**3)*100)):.1f}%")
        
        st.html(f"""
            <div class="glass-card" style="border: 2px solid #58a6ff; margin-top:20px;">
                <span class="val-title">Theoretische Standzeit</span>
                <span class="val-main" style="color:#58a6ff; font-size:3rem;">~ {pred_cycles} Zyklen</span>
                <p style="color:#8b949e; font-size:0.9rem;">Bei {l_wear_cycle:.3f}% Verschleiß pro Bohrung</p>
            </div>
        """)

# --- RUNTIME ---
st.divider()
b1, b2 = st.columns(2)
if b1.button("▶ START / PAUSE", use_container_width=True): s['active'] = not s['active']; st.rerun()
if b2.button("🔄 RESET WERKZEUG", use_container_width=True):
    st.session_state.twin = {'zyklus': 0.0, 'zyklen_anzahl': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False, 'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0, 'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0, 'seed': np.random.RandomState(42)}
    st.rerun()

if s['active']:
    time.sleep(max(0.01, taktzeit/1000))
    st.rerun()
