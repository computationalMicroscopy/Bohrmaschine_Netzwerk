import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-READABILITY STYLING ---
st.set_page_config(layout="wide", page_title="KI-Zerspanungslabor Pro V3.2", page_icon="⚙️")

# Reines CSS/HTML wird jetzt via st.html geladen -> Schützt vor Markdown-Konflikten
st.html("""
    <style>
    /* Globaler Präsentations-Darkmode */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Extreme Lesbarkeit für Widgets & Regler */
    label, .stSlider, .stSelectbox, .stToggle { 
        font-size: 1.35rem !important; 
        font-weight: 700 !important; 
        color: #f0f6fc !important;
    }
    .stMarkdown p { font-size: 1.25rem !important; line-height: 1.6; }
    
    .main-title {
        font-size: 3.0rem; font-weight: 800; color: #58a6ff;
        margin-bottom: 25px; text-align: center; border-bottom: 3px solid #30363d; padding-bottom: 15px;
    }
    .glass-card {
        background: #161b22; border: 2px solid #30363d;
        border-radius: 14px; padding: 24px; margin-bottom: 12px; text-align: center;
    }
    .val-title { font-size: 1.15rem; color: #8b949e; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 800; color: #f0f6fc; margin-top: 5px; display: block; }
    
    /* XAI Diagnose-Karten */
    .xai-container { height: 620px; overflow-y: auto; padding-right: 5px; }
    .xai-card {
        background: #1f242c; border-left: 7px solid #58a6ff;
        padding: 20px; border-radius: 10px; margin-bottom: 14px;
    }
    .xai-feature-row { display: flex; justify-content: space-between; font-size: 1.2rem; color: #c9d1d9; margin-top: 6px; font-weight: 500;}
    .xai-bar-bg { background: #21262d; height: 12px; width: 100%; border-radius: 6px; margin-bottom: 8px; }
    .xai-bar-fill { background: #58a6ff; height: 12px; border-radius: 6px; }
    .reason-text { color: #f0f6fc; font-size: 1.35rem; margin-top: 6px; font-weight: 700; }
    .sensor-snapshot { font-size: 1.1rem; color: #8b949e; margin-top: 6px; font-family: monospace; border-bottom: 1px solid #30363d; padding-bottom: 6px;}
    .action-text { color: #ff7b72; font-weight: bold; font-size: 1.2rem; margin-top: 8px; border-top: 1px solid #30363d; padding-top: 8px; }
    .diag-badge { background: #388bfd; color: #ffffff; padding: 5px 10px; border-radius: 4px; font-size: 14px; font-weight: 800; }
    
    .emergency-alert {
        background: #da3633; color: white; padding: 20px; border-radius: 10px; 
        font-weight: bold; text-align: center; margin-bottom: 25px; font-size: 1.6rem;
    }

    /* --- ADVANCED REALISTIC DRILL ANIMATION KEYFRAMES --- */
    @keyframes twist_effect {
        0% { background-position: 0px 0px; }
        100% { background-position: 0px 120px; }
    }
    @keyframes severe_shake {
        0% { transform: translate(1.5px, 0.5px) rotate(0.1deg); }
        100% { transform: translate(-1.5px, -0.5px) rotate(-0.1deg); }
    }
    </style>
""")

st.html('<div class="main-title">🚀 Next-Gen KI-Zerspanungslabor & XAI-Plattform</div>')

# --- 2. XAI ROOT-CAUSE ENGINE ---
def get_expert_diagnostics(top_reason, current_vals, settings, integrity):
    vc, f, d = settings['vc'], settings['f'], settings['d']
    
    mapping = {
        "Mechanische Torsions-Überlast": {
            "diag": "CRITICAL TORQUE", 
            "exp": f"Das Drehmoment ({current_vals['M']:.1f} Nm) überlastet den Schaftquerschnitt.", 
            "act": f"ABHILFE: Vorschub f auf {f*0.7:.2f} mm/U senken! Das Reduzieren von vc bringt hier physikalisch keine Entlastung."
        },
        "Thermische Gefüge-Erweichung (Schneidkanten-Härteverlust durch extreme Hitze)": {
            "diag": "THERMAL OVERLOAD", 
            "exp": f"Schnittkantentemperatur ({current_vals['T']:.0f}°C) liegt im kritischen Bereich. Der Schneidstoff verliert durch Gefügeänderung seine Härte und verformt sich plastisch.", 
            "act": f"ABHILFE: Schnittgeschwindigkeit vc um 25% auf {int(vc*0.75)} m/min reduzieren, um Reibungsleistung zu senken."
        },
        "Regeneratives Rattern (Resonanz)": {
            "diag": "RESONANT CHATTER", 
            "exp": f"Starke Vibrationsamplitude ({current_vals['V']:.1f} mm/s) zerstört die Schneidkantenmikrogeometrie.", 
            "act": f"ABHILFE: Drehzahl-Shift erfordern! Ändere vc um ±15% ({int(vc*0.85)} / {int(vc*1.15)} m/min), um Harmonische zu brechen."
        },
        "Kühlungs-Abriss (Adhäsion)": {
            "diag": "TRIBOLOGY FAILURE", 
            "exp": "Schmierung kollabiert. Spanflächenreibung steigt sprunghaft an, Spanfestklebung droht.", 
            "act": "NOT-STOPP: KSS-Zuleitung und Pumpendruck prüfen. Sofortiger Vorschubstopp zwingend."
        },
        "Axiale Schaft-Knickung (Stabilitätsversagen des Bohrers durch zu hohe Vorschubkraft)": {
            "diag": "AXIAL BUCKLED", 
            "exp": f"Die Vorschubkraft ({current_vals['F']:.0f} N) überschreitet die elastische Stabilitätsgrenze (Euler-Knickung). Der lange Bohrschaft weicht elastisch seitlich aus und bricht schlagartig.", 
            "act": f"ABHILFE: Vorschub f sofort halbieren! Gefahr von irreversiblem Achsversatz oder Bohrer-Splitterung."
        },
        "Normaler Standzeit-Abrieb": {
            "diag": "ABRASIVE WEAR", 
            "exp": f"Fortgeschrittener mechanischer Verschleiß der Freifläche (Restintegrität {integrity:.1f}%).", 
            "act": "GEPLANTER WECHSEL: Werkzeug hat das Ende des Standzeitfensters erreicht. Demnächst tauschen."
        }
    }
    res = mapping.get(top_reason, {"diag": "NOMINAL PROCESS", "exp": "Prozessparameter im grünen Bereich.", "act": "Keine Korrekturmaßnahmen notwendig."})
    res["snapshot"] = f"M: {current_vals['M']:.1f}Nm | T: {current_vals['T']:.0f}°C | V: {current_vals['V']:.1f}mm/s | F_f: {current_vals['F']:.0f}N"
    return res

# --- 3. INITIALISIERUNG & STATE-MACHINE ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'zyklus': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }

s = st.session_state.twin

MATERIALIEN = {
    "Baustahl (1.0037)": {"kc1.1": 1800, "mc": 0.25, "wear_factor": 0.02, "t_crit": 450}, 
    "Vergütungsstahl (1.7225)": {"kc1.1": 2100, "mc": 0.24, "wear_factor": 0.05, "t_crit": 550}, 
    "Titanlegierung (3.7165)": {"kc1.1": 2500, "mc": 0.23, "wear_factor": 0.16, "t_crit": 650},
    "Edelstahl (1.4301)": {"kc1.1": 2300, "mc": 0.22, "wear_factor": 0.12, "t_crit": 600}
}

LABELS = [
    "Mechanische Torsions-Überlast", 
    "Thermische Gefüge-Erweichung (Schneidkanten-Härteverlust durch extreme Hitze)", 
    "Regeneratives Rattern (Resonanz)", 
    "Kühlungs-Abriss (Adhäsion)", 
    "Axiale Schaft-Knickung (Stabilitätsversagen des Bohrers durch zu hohe Vorschubkraft)", 
    "Normaler Standzeit-Abrieb"
]

# --- 4. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Live-Prozessparameter")
    mat_name = st.selectbox("Ausgewählter Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    
    vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 30, 350, 100)
    f = st.slider("Vorschub pro Umdrehung f [mm/U]", 0.05, 0.60, 0.15)
    d = st.slider("Bohrer-Durchmesser d [mm]", 5.0, 32.0, 12.0)
    kuehlung = st.toggle("Kühlschmierstoff (KSS) active", value=True)
    
    st.divider()
    st.header("📡 Sensor-Kalibrierung & Skalierung")
    sensor_temp_gain = st.slider("Temperatursensor-Empfindlichkeit", 0.5, 2.5, 1.0, step=0.1)
    sensor_vibr_gain = st.slider("Vibrationssensor-Verstärkung (Gain)", 0.5, 3.0, 1.0, step=0.1)
    
    st.divider()
    st.header("🎛️ Signalstörungen")
    noise_level = st.slider("Rausch-Amplitude (Vibration)", 0.1, 2.0, 0.5)
    
    st.divider()
    schrittweite = st.number_input("Simulationsschritte pro Takt", 5, 100, 20)
    taktzeit = st.select_slider("Aktualisierungsintervall (ms)", options=[500, 200, 100, 0], value=100)

# --- 5. PHYSIK-ENGINE (LIVE-PROZESS) ---
n = (vc * 1000) / (np.pi * d) if d > 0 else 0
s['drehzahl'] = n

if s['active'] and not s['broken'] and not s['stall']:
    s['zyklus'] += schrittweite
    
    h = (f / 2.0)
    kc = m['kc1.1'] * (h ** -m['mc'])
    
    s['drehmoment'] = (f * (d**2) * kc) / 8000.0  
    s['vorschubkraft'] = (0.5 * d * f * kc) * 1.3  
    s['leistung'] = (s['drehmoment'] * n) / 9550.0  
    
    crit_torque = 0.12 * (d ** 3) 
    crit_force = 320 * (d ** 2)
    
    p_friction_watts = s['drehmoment'] * (n * 2 * np.pi / 60.0)
    kss_factor = 4.2 if kuehlung else 0.9  
    t_target = 22.0 + ((p_friction_watts * 0.22) / kss_factor) * sensor_temp_gain
    s['thermik'] += (t_target - s['thermik']) * 0.15 
    
    chatter_trigger = 1.2 + (s['drehmoment'] * 0.12) if (int(n) % 400 < 60) else 0.25
    s['vibration'] = max(0.1, (chatter_trigger + s['seed'].normal(0, 0.1) * noise_level) * sensor_vibr_gain)
    
    norm_torque = s['drehmoment'] / crit_torque
    norm_force = s['vorschubkraft'] / crit_force
    norm_temp = s['thermik'] / m['t_crit']
    norm_vibr = s['vibration'] / 8.0
    kss_loss = 1.0 if not kuehlung else 0.0
    norm_wear = (100.0 - s['integritaet']) / 100.0
    
    weights = [3.0, 3.5, 2.5, 4.0, 3.0, 3.5]
    scores = np.array([
        norm_torque * weights[0], norm_temp * weights[1], norm_vibr * weights[2],
        kss_loss * weights[3], norm_force * weights[4], norm_wear * weights[5]
    ])
    
    exp_s = np.exp(scores - np.max(scores))
    probabilities = (exp_s / exp_s.sum()) * 100
    evidenz_list = sorted(zip(LABELS, probabilities), key=lambda x: x[1], reverse=True)
    
    max_stress = max([norm_torque, norm_force, norm_temp, norm_vibr])
    combined_risk_score = max_stress + (norm_wear ** 2.0) * 0.95
    
    if combined_risk_score < 0.75:
        s['risk'] = combined_risk_score * 0.20  
    else:
        s['risk'] = 0.15 + (combined_risk_score - 0.75) * 2.5
    s['risk'] = np.clip(s['risk'], 0.01, 0.99)
    
    v_factor = (vc / 120.0) ** 1.6
    thermal_accelerator = np.exp(max(0.0, s['thermik'] - m['t_crit']) / 25.0)
    s['abrasion'] += (m['wear_factor'] * v_factor * f * thermal_accelerator) * (schrittweite / 10.0)
    
    fatigue = 0.0
    if norm_torque > 0.85: fatigue += (norm_torque - 0.85) ** 2
    if norm_force > 0.85: fatigue += (norm_force - 0.85) ** 2
    
    total_wear_increment = ((s['abrasion'] * 0.002) + fatigue) * schrittweite
    s['integritaet'] = max(0.0, s['integritaet'] - total_wear_increment)
    
    if s['leistung'] > 7.5:
        s['stall'] = True
        s['active'] = False
    
    if s['integritaet'] <= 0.0 or norm_torque > 1.15 or norm_force > 1.2 or (norm_temp > 1.1 and s['drehmoment'] > crit_torque * 0.5):
        s['broken'] = True
        s['active'] = False
        s['integritaet'] = 0.0
    
    exp_report = get_expert_diagnostics(evidenz_list[0][0], {'M': s['drehmoment'], 'T': s['thermik'], 'V': s['vibration'], 'F': s['vorschubkraft']}, {'vc': vc, 'f': f, 'd': d}, s['integritaet'])
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'info': exp_report, 'evidenz': evidenz_list})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration'], 'p': s['leistung'], 'm': s['drehmoment'], 'f': s['vorschubkraft']})

# --- 6. HIGH-REALISMUS VISUELLES COCKPIT (ST.HTML SCHÜTZT VOR BREAKS) ---
if s['broken']:
    st.html('<div class="emergency-alert">💥 STRUKTURELLER WERKZEUGBRUCH! Schaft durch mechanische Überlast komplett zerstört.</div>')
if s['stall']:
    st.html('<div class="emergency-alert">⚠️ MOTOR-STALL: Leistungsaufnahme überschreitet maximales Drehmoment der Spindel (7.5 kW).</div>')

col_animation, col_metrics = st.columns([1.2, 4])

with col_animation:
    t_val = s['thermik']
    if t_val < 150:
        target_color = "#555555"
        glow_box = "none"
    elif t_val < 350:
        factor = (t_val - 150) / 200
        r = int(85 + (120 - 85) * factor)
        g = int(85 + (80 - 85) * factor)
        b = int(85 + (40 - 85) * factor)
        target_color = f"rgb({r},{g},{b})"
        glow_box = f"0 0 {int(5 + 10*factor)}px rgba(255, 106, 0, 0.4)"
    else:
        factor = min(1.0, (t_val - 350) / 350)
        r = int(200 + (55 * factor))
        g = int(50 * factor)
        target_color = f"rgb({r}, {g}, 0)"
        glow_box = f"0 0 {int(15 + 25*factor)}px rgba({r}, {g}, 0, 0.9)"

    glow_style = f"background: {target_color}; box-shadow: {glow_box};"

    if s['broken']:
        anim_spin, anim_shake = "none", "none"
        body_render = """
        <div style="width: 45px; height: 60px; background: #444; margin-left: 10px; transform: rotate(-20deg); border-bottom: 2px dashed red;"></div>
        <div style="width: 45px; height: 60px; background: #333; margin-top: 15px; margin-left: -20px; transform: rotate(45deg); clip-path: polygon(0% 0%, 100% 0%, 50% 100%);"></div>
        """
        status_label = "<span style='color:#ff7b72; font-weight:900;'>CRASH / BRUCH</span>"
    else:
        spin_duration = f"{max(0.02, 45.0 / (s['drehzahl'] + 1)):.3f}s" if s['active'] else "0s"
        anim_spin = f"twist_effect {spin_duration} linear infinite" if s['active'] else "none"
        shake_duration = f"{max(0.01, 0.1 / (s['vibration'] + 0.01)):.3f}s"
        anim_shake = f"severe_shake {shake_duration} infinite alternate" if s['active'] or s['stall'] else "none"
        
        body_render = f"""
        <div style="animation: {anim_spin}; width: 45px; height: 110px; background: linear-gradient(120deg, #777 20%, #222 35%, #888 50%, #222 65%, #666 80%); background-size: 45px 40px; box-shadow: inset 2px 0 10px rgba(0,0,0,0.5); border-radius: 0 0 2px 2px;"></div>
        <div style="{glow_style} width: 45px; height: 22px; clip-path: polygon(0% 0%, 100% 0%, 50% 100%); margin-top: -1px; transition: all 0.2s;"></div>
        """
        status_label = "<span style='color:#2ea44f; font-weight:900;'>ROTATION LIVE</span>" if s['active'] else "<span style='color:#8b949e;'>STANDBY</span>"
        if s['stall']: status_label = "<span style='color:#e3b341; font-weight:900;'>STALL / BLOCKIERT</span>"

    # Komplett über st.html gekapselt -> verzieht sich niemals im Layout
    st.html(f"""
        <div class="glass-card" style="padding: 16px; height: 100%; min-height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; background: #12161f;">
            <span class="val-title" style="margin-bottom: 10px; color: #58a6ff;">Spindel-Monitor</span>
            <div style="width: 75px; height: 40px; background: linear-gradient(to right, #333, #555, #333); border-radius: 6px 6px 0 0; border: 1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"></div>
            <div style="animation: {anim_shake}; width: 100%; display: flex; flex-direction: column; align-items: center;">
                {body_render}
            </div>
            <div style="margin-top: 15px; font-size: 1.15rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.7px; background: #161b22; padding: 4px 12px; border-radius: 6px; border: 1px solid #30363d; text-align:center; min-width:140px;">{status_label}</div>
        </div>
    """)

with col_metrics:
    c0, c1, c2, c3, c4, c5 = st.columns(6)
    c0.html(f'<div class="glass-card"><span class="val-title">Zyklen</span><br><span class="val-main">{s["zyklus"]}</span></div>')
    c1.html(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#2ea44f">{s["integritaet"]:.1f}%</span></div>')
    c2.html(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#f85149">{s["risk"]:.1%}</span></div>')
    c3.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:#ff7b72">{s["thermik"]:.0f}°C</span></div>')
    c4.html(f'<div class="glass-card"><span class="val-title">Schwingung</span><br><span class="val-main" style="color:#a371f7">{s["vibration"]:.2f} mm/s</span></div>')
    c5.html(f'<div class="glass-card"><span class="val-title">Drehmoment</span><br><span class="val-main" style="color:#e3b341">{s["drehmoment"]:.1f} Nm</span></div>')

# --- 7. TABS: LIVE TRENDS VS SZENARIEN-LABOR ---
t1, t2 = st.tabs(["📈 Live-Prozessüberwachung & Oszilloskop", "🔬 Interaktives Was-Wäre-Wenn Szenarien-Labor"])

with t1:
    col_graph, col_log = st.columns([2, 1])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                subplot_titles=("Strukturelle Werkzeugintegrität (%)", "Kinetische Lasten: Drehmoment (Nm) & Axialkraft (N)", "Prozessdynamik: Schwingung (mm/s) & Temperatur (°C)"))
            
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#2ea44f', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['m'], line=dict(color='#e3b341', width=2.5), name="Moment"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['f']/100.0, line=dict(color='#1f6feb', width=2, dash='dash'), name="Axialkraft/100"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#a371f7', width=2.5), name="Vibration"), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#ff7b72', width=2.5), name="Temp"), 3, 1)
            
            fig.update_layout(height=600, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Simulation starten, um Telemetriedaten aufzuzeichnen.")
            
    with col_log:
        st.markdown("### 👁️ KI-Ursachendiagnose (XAI)")
        if s['logs']:
            html_str = '<div class="xai-container">'
            for l in s['logs'][:10]:
                bars = "".join([f'<div class="xai-feature-row"><span>{e[0]}</span><span>{e[1]:.1f}%</span></div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{e[1]}%"></div></div>' for e in l['evidenz'][:3]])
                html_str += f"""
                <div class="xai-card">
                    <div style="display:flex; justify-content:between; align-items:center; margin-bottom:6px;">
                        <span class="diag-badge">{l['info']['diag']}</span>
                        <span style="font-size:11px; color:#8b949e; margin-left:auto;">{l['zeit']}</span>
                    </div>
                    <div class="reason-text">{l['info']['exp']}</div>
                    <div class="sensor-snapshot">{l['info']['snapshot']}</div>
                    <div style="margin-top:6px;">{bars}</div>
                    <div class="action-text">{l['info']['act']}</div>
                </div>"""
            html_str += '</div>'
            st.html(html_str)

# --- 8. WAS-WÄRE-WENN LABOR ---
with t2:
    st.markdown("### 🧪 Labor-Simulationsraum für hypothetische Grenzbereiche")
    
    col_inputs, col_outputs = st.columns([1, 1])
    
    with col_inputs:
        st.subheader("⚙️ Hypothetische Parameter-Konfiguration")
        lab_mat = st.selectbox("Zu bearbeitender Labor-Werkstoff", list(MATERIALIEN.keys()), key="lab_mat")
        lm = MATERIALIEN[lab_mat]
        
        lab_vc = st.slider("Eingestellte Schnittgeschwindigkeit vc [m/min]", 30, 350, 100, key="lab_vc")
        lab_f = st.slider("Gewählter Vorschub f [mm/U]", 0.05, 0.60, 0.15, key="lab_f")
        lab_d = st.slider("Bohrer-Nenndurchmesser d [mm]", 5.0, 32.0, 12.0, key="lab_d")
        lab_kss = st.toggle("Kühlschmierstoff-Zufuhr aktiv (KSS)", value=True, key="lab_kss")
        
        st.divider()
        st.subheader("⏳ Simulierter Werkzeugzustand")
        lab_integ = st.slider("Aktuelle Werkzeug-Restlebensdauer / Integrität [%]", 0.0, 100.0, 100.0, key="lab_integ")
        lab_vibr_override = st.slider("Künstlich überlagerte Vibrationsamplitude [mm/s]", 0.1, 15.0, 0.4, key="lab_vibr")

    with col_outputs:
        st.subheader("📊 Berechnete physikalische Zielgrößen")
        
        l_n = (lab_vc * 1000) / (np.pi * lab_d) if lab_d > 0 else 0
        l_h = (lab_f / 2.0)
        l_kc = lm['kc1.1'] * (l_h ** -lm['mc'])
        
        l_torque = (lab_f * (lab_d**2) * l_kc) / 8000.0
        l_force = (0.5 * lab_d * lab_f * l_kc) * 1.3
        l_power = (l_torque * l_n) / 9550.0
        
        l_p_friction = l_torque * (l_n * 2 * np.pi / 60.0)
        l_kss_fac = 4.2 if lab_kss else 0.9  
        l_temp = (22.0 + (l_p_friction * 0.22) / l_kss_fac) * sensor_temp_gain
        
        c_t = 0.12 * (lab_d ** 3)
        c_f = 320 * (lab_d ** 2)
        
        l_norm_torque = l_torque / c_t
        l_norm_force = l_force / c_f
        l_norm_temp = l_temp / lm['t_crit']
        l_norm_vibr = (lab_vibr_override * sensor_vibr_gain) / 8.0
        l_kss_loss = 1.0 if not lab_kss else 0.0
        l_norm_wear = (100.0 - lab_integ) / 100.0
        
        l_weights = [3.0, 3.5, 2.5, 4.0, 3.0, 3.5]
        l_scores = np.array([
            l_norm_torque * l_weights[0], l_norm_temp * l_weights[1], l_norm_vibr * l_weights[2],
            l_kss_loss * l_weights[3], l_norm_force * l_weights[4], l_norm_wear * l_weights[5]
        ])
        
        l_exp = np.exp(l_scores - np.max(l_scores))
        l_probs = (l_exp / l_exp.sum()) * 100
        l_evidenz = sorted(zip(LABELS, l_probs), key=lambda x: x[1], reverse=True)
        
        l_max_stress = max([l_norm_torque, l_norm_force, l_norm_temp, l_norm_vibr])
        l_combined_score = l_max_stress + (l_norm_wear ** 2.0) * 0.95
        
        if l_combined_score < 0.75:
            lab_risk = l_combined_score * 0.20
        else:
            lab_risk = 0.15 + (l_combined_score - 0.75) * 2.5
        lab_risk = np.clip(lab_risk, 0.01, 0.99)
        
        lc1, lc2 = st.columns(2)
        lc1.html(f'<div class="glass-card"><span class="val-title">Errechnetes Drehmoment</span><br><span class="val-main" style="color:#e3b341">{l_torque:.1f} Nm</span><br><span style="font-size:1.1rem;color:#8b949e;font-weight:bold;">Bruch-Limit: {c_t:.1f} Nm</span></div>')
        lc2.html(f'<div class="glass-card"><span class="val-title">Errechnete Temperatur</span><br><span class="val-main" style="color:#ff7b72">{l_temp:.0f} °C</span><br><span style="font-size:1.1rem;color:#8b949e;font-weight:bold;">Werkstoff-Erweichungs-Limit: {lm["t_crit"]} °C</span></div>')
        
        lc3, lc4 = st.columns(2)
        lc3.html(f'<div class="glass-card"><span class="val-title">Erwartete Spindellast</span><br><span class="val-main" style="color:#58a6ff">{l_power:.2f} kW</span><br><span style="font-size:1.1rem;color:#8b949e;font-weight:bold;">Max. Motorleistung: 7.5 kW</span></div>')
        lc4.html(f'<div class="glass-card"><span class="val-title">Errechnete Vorschubkraft</span><br><span class="val-main" style="color:#1f6feb">{l_force:.0f} N</span><br><span style="font-size:1.1rem;color:#8b949e;font-weight:bold;">Knick-Limit: {c_f:.0f} N</span></div>')
        
        st.html(f"""
            <div class="glass-card" style="border: 3px solid #58a6ff; margin-top:15px; background: #1f242c;">
                <span class="val-title" style="font-size:1.35rem;">Präzisiertes KI-Bruchrisiko:</span><br>
                <span class="val-main" style="color:#ff7b72; font-size:3.4rem; margin-top:8px;">{lab_risk:.1%}</span>
            </div>
        """)
        
        st.markdown("### 🔬 Prädizierte KI-Ursachengewichtung:")
        html_bars = ""
        for name, prob in l_evidenz[:4]:
            html_bars += f"""
            <div class="xai-feature-row" style="margin-top:12px;"><span>{name}</span><span style="font-weight:bold;">{prob:.1f}%</span></div>
            <div class="xai-bar-bg" style="height:12px;"><div class="xai-bar-fill" style="width:{prob}%; height:12px;"></div></div>
            """
        st.html(html_bars)

# --- 9. RUNTIME CONTROLS ---
st.divider()
b1, b2 = st.columns(2)
if b1.button("▶ SIMULATION STARTEN / PAUSIEREN", use_container_width=True):
    st.session_state.twin['active'] = not st.session_state.twin['active']
if b2.button("🔄 NEUES WERKZEUG EINSPANNEN (RESET)", use_container_width=True):
    st.session_state.twin = {
        'zyklus': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }
    st.rerun()

if s['active']:
    if taktzeit > 0: time.sleep(taktzeit / 1000)
    st.rerun()
