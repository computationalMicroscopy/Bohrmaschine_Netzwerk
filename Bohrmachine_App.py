import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-END INDUSTRIAL STYLING ---
st.set_page_config(layout="wide", page_title="KI-Zerspanungs-Plattform TwinPro V5.0", page_icon="⚙️")

st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;800&display=swap');
    
    /* Globaler Next-Gen Darkmode */
    .stApp { 
        background-color: #06090e; 
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Regler & Control-UX */
    label, .stSlider, .stSelectbox, .stToggle { 
        font-size: 1.25rem !important; 
        font-weight: 600 !important; 
        color: #f0f6fc !important;
    }
    .stMarkdown p { font-size: 1.15rem !important; line-height: 1.6; }
    
    .main-title {
        font-size: 2.8rem; font-weight: 800; color: #f0f6fc;
        text-align: center; margin-bottom: 30px; padding-bottom: 20px;
        border-bottom: 1px solid rgba(240, 246, 252, 0.1);
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* High-Tech Cockpit Cards */
    .glass-card {
        background: rgba(16, 22, 30, 0.85); 
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 12px; padding: 20px; margin-bottom: 15px; text-align: center;
        box-shadow: 0 4px 25px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.5);
        box-shadow: 0 4px 30px rgba(88, 166, 255, 0.15);
    }
    .val-title { font-size: 1.05rem; color: #8b949e; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 800; color: #f0f6fc; margin-top: 5px; display: block; }
    
    /* XAI Diagnose-Karten */
    .xai-container { height: 600px; overflow-y: auto; padding-right: 5px; }
    .xai-card {
        background: #0d1117; border-left: 6px solid #58a6ff;
        padding: 18px; border-radius: 8px; margin-bottom: 12px;
        border-top: 1px solid rgba(255,255,255,0.02);
    }
    .xai-feature-row { display: flex; justify-content: space-between; font-size: 1.1rem; color: #c9d1d9; margin-top: 6px; font-weight: 500;}
    .xai-bar-bg { background: #21262d; height: 8px; width: 100%; border-radius: 4px; margin-bottom: 8px; overflow:hidden;}
    .xai-bar-fill { background: linear-gradient(90deg, #58a6ff, #1f6feb); height: 100%; border-radius: 4px; }
    .reason-text { color: #f0f6fc; font-size: 1.25rem; margin-top: 6px; font-weight: 700; }
    .sensor-snapshot { font-size: 1.0rem; color: #8b949e; margin-top: 6px; font-family: 'JetBrains Mono', monospace; border-bottom: 1px solid #30363d; padding-bottom: 6px;}
    .action-text { color: #ff7b72; font-weight: bold; font-size: 1.15rem; margin-top: 8px; border-top: 1px solid rgba(48, 54, 61, 0.5); padding-top: 8px; }
    .diag-badge { background: #23426f; color: #58a6ff; border: 1px solid #388bfd; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;}
    
    .emergency-alert {
        background: linear-gradient(90deg, #da3633, #8a1f1d); color: white; padding: 20px; border-radius: 10px; 
        font-weight: 800; text-align: center; margin-bottom: 25px; font-size: 1.5rem;
        box-shadow: 0 0 30px rgba(218, 54, 51, 0.4);
        border: 1px solid #f85149;
    }

    /* --- CINEMATIC ANIMATION KEYFRAMES --- */
    @keyframes helical_spin {
        0% { background-position: 0px 0px, 0px 0px; }
        100% { background-position: 0px -180px, 0px 0px; }
    }
    @keyframes tool_feed {
        0% { transform: translateY(-15px); }
        45% { transform: translateY(22px); } /* Maximale Schnitttiefe */
        55% { transform: translateY(22px); }
        100% { transform: translateY(-15px); } /* Spanentleerung / Rückzug */
    }
    @keyframes industrial_shake {
        0% { transform: translate(0.4px, 0.4px) rotate(0.04deg); }
        50% { transform: translate(-0.8px, 0.2px) rotate(-0.04deg); }
        100% { transform: translate(0.4px, -0.6px) rotate(0.02deg); }
    }
    @keyframes chip_spray_left {
        0% { transform: translate(0, 0) scale(1.2) rotate(0deg); opacity: 1; }
        80% { opacity: 0.9; }
        100% { transform: translate(-85px, -50px) scale(0.1) rotate(-540deg); opacity: 0; }
    }
    @keyframes chip_spray_right {
        0% { transform: translate(0, 0) scale(1.2) rotate(0deg); opacity: 1; }
        80% { opacity: 0.9; }
        100% { transform: translate(85px, -50px) scale(0.1) rotate(540deg); opacity: 0; }
    }
    @keyframes smoke_rise {
        0% { transform: translate(-50%, 0px) scale(0.5); opacity: 0; }
        30% { opacity: 0.4; filter: blur(4px); }
        100% { transform: translate(-50%, -75px) scale(3.5); opacity: 0; filter: blur(8px); }
    }
    @keyframes kss_flood_left {
        0% { transform: translate(-45px, -65px) scaleY(1) rotate(40deg); opacity: 0.7; }
        100% { transform: translate(-3px, -2px) scaleX(0.3) rotate(40deg); opacity: 1; }
    }
    @keyframes kss_flood_right {
        0% { transform: translate(45px, -65px) scaleY(1) rotate(-40deg); opacity: 0.7; }
        100% { transform: translate(3px, -2px) scaleX(0.3) rotate(-40deg); opacity: 1; }
    }
    @keyframes kss_mist {
        0% { transform: translate(-50%, -10px) scale(0.8); opacity: 0.3; }
        50% { opacity: 0.6; }
        100% { transform: translate(-50%, -30px) scale(1.8); opacity: 0; }
    }
    @keyframes led_pulse {
        0% { box-shadow: 0 0 8px var(--led-color), inset 0 0 4px var(--led-color); }
        50% { box-shadow: 0 0 22px var(--led-color), inset 0 0 10px var(--led-color); }
        100% { box-shadow: 0 0 8px var(--led-color), inset 0 0 4px var(--led-color); }
    }
    @keyframes strobe_crit {
        0%, 100% { background: #ff0000; box-shadow: 0 0 25px #ff0000; }
        50% { background: #200000; box-shadow: 0 0 2px #200000; }
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
    kuehlung = st.toggle("Kühlschmierstoff (KSS) aktiv", value=True)
    
    st.divider()
    st.header("📡 Sensor-Kalibrierung & Skalierung")
    sensor_temp_gain = st.slider("Temperatursensor-Empfindlichkeit", 0.5, 2.5, 1.0, step=0.1)
    sensor_vibr_gain = st.slider("Vibrationssensor-Verstärkung (Gain)", 0.5, 3.0, 1.0, step=0.1)
    
    st.divider()
    st.header("🎛️ Signalstörungen")
    noise_level = st.slider("Rausch-Amplitude (Vibration)", 0.1, 2.0, 0.5)
    
    st.divider()
    schrittweite = st.number_input("Zeitskalierungsfaktor", 1, 20, 5)
    taktzeit = st.select_slider("Aktualisierungsintervall (ms)", options=[500, 200, 100, 0], value=100)

# --- 5. PHYSIK-ENGINE ---
n = (vc * 1000) / (np.pi * d) if d > 0 else 0
s['drehzahl'] = n

if s['active'] and not s['broken'] and not s['stall']:
    dt = (taktzeit / 1000.0 if taktzeit > 0 else 0.05) * schrittweite
    
    # Präzise Zyklen-Synchronisation basierend auf der CSS-Animationsdauer (2.5 Sekunden pro Loch)
    altes_zeitfenster = int(s['zyklus'] / 2.5)
    s['zyklus'] += dt 
    neues_zeitfenster = int(s['zyklus'] / 2.5)
    if neues_zeitfenster > altes_zeitfenster:
        s['zyklen_anzahl'] += 1  # Ein kompletter Hub beendet -> Zähler inkrementiert!
    
    h = (f / 2.0)
    kc = m['kc1.1'] * (h ** -m['mc'])
    
    s['drehmoment'] = (f * (d**2) * kc) / 8000.0  
    s['vorschubkraft'] = (0.5 * d * f * kc) * 1.3  
    s['leistung'] = (s['drehmoment'] * n) / 9550.0  
    
    crit_torque = 0.12 * (d ** 3) 
    crit_force = 320 * (d ** 2)
    
    p_friction_watts = s['drehmoment'] * (n * 2 * np.pi / 60.0)
    kss_factor = 4.5 if kuehlung else 0.85  
    t_target = 22.0 + ((p_friction_watts * 0.20) / kss_factor) * sensor_temp_gain
    s['thermik'] += (t_target - s['thermik']) * 0.20 
    
    chatter_trigger = 1.0 + (s['drehmoment'] * 0.10) if (int(n) % 400 < 60) else 0.20
    s['vibration'] = max(0.1, (chatter_trigger + s['seed'].normal(0, 0.08) * noise_level) * sensor_vibr_gain)
    
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
        s['risk'] = combined_risk_score * 0.18  
    else:
        s['risk'] = 0.15 + (combined_risk_score - 0.75) * 2.3
    s['risk'] = np.clip(s['risk'], 0.01, 0.99)
    
    v_factor = (vc / 120.0) ** 1.5
    thermal_accelerator = np.exp(max(0.0, s['thermik'] - m['t_crit']) / 30.0)
    s['abrasion'] += (m['wear_factor'] * v_factor * f * thermal_accelerator) * dt
    
    fatigue = 0.0
    if norm_torque > 0.85: fatigue += (norm_torque - 0.85) ** 2
    if norm_force > 0.85: fatigue += (norm_force - 0.85) ** 2
    
    total_wear_increment = ((s['abrasion'] * 0.02) + fatigue * 0.5) * dt * 10.0
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

# --- 6. CRASH-DETEKTION ---
if s['broken']:
    st.html('<div class="emergency-alert">💥 STRUKTURELLER WERKZEUGBRUCH! Schaft durch mechanische Überlast komplett zerstört.</div>')
if s['stall']:
    st.html('<div class="emergency-alert">⚠️ MOTOR-STALL: Leistungsaufnahme übersteigt maximales Drehmoment der Spindel (7.5 kW).</div>')

col_animation, col_metrics = st.columns([1.5, 4])

with col_animation:
    t_val = s['thermik']
    integ_val = s['integritaet']
    
    # 1. Bestimmung des LED-Statusring-Farbwerts
    if s['broken'] or s['stall']:
        led_style = "animation: strobe_crit 0.2s infinite;"
    elif not s['active']:
        led_style = "background: #555; box-shadow: 0 0 5px #333; --led-color: #555;"
    else:
        if s['risk'] > 0.7:
            color, hex_val = "rgba(255, 68, 0, 1)", "#ff4400"
        elif kuehlung:
            color, hex_val = "rgba(0, 180, 255, 1)", "#00b4ff"
        else:
            color, hex_val = "rgba(46, 164, 79, 1)", "#2ea44f"
        led_style = f"background: {hex_val}; --led-color: {color}; animation: led_pulse 1s infinite ease-in-out;"

    # 2. Thermisches Glühen der Schneidenspitze
    if t_val < 150:
        tip_base = "#555555"
        glow_effect = "rgba(0,0,0,0)"
    elif t_val < 380:
        factor = (t_val - 150) / 230
        tip_base = f"rgb({int(85+80*factor)},{int(85-20*factor)},{int(85-65*factor)})"
        glow_effect = f"0 8px 20px rgba(255, 90, 0, {0.3 * factor})"
    else:
        factor = min(1.0, (t_val - 380) / 320)
        tip_base = f"rgb({int(165+90*factor)}, {int(65*(1.0-factor))}, 0)"
        glow_effect = f"0 8px 30px rgba({int(165+90*factor)}, {int(65*(1.0-factor))}, 0, {0.5 + 0.4 * factor})"

    # 3. Mechanischer Verschleiß & Brandflecken-Überlagerung
    wear_factor = (100.0 - integ_val) / 100.0
    tip_color = f"linear-gradient(to bottom, #444 0%, {tip_base} 65%, rgba(15,15,15,{wear_factor:.2f}) 100%)" if not s['broken'] else "#1a1a1a"

    # 4. Glühende Verformungszone des Werkstücks (Eintrittsloch)
    surface_glow = f"rgba(255, 60, 0, {min(0.9, (t_val-100)/500)})" if t_val > 100 else "rgba(0,0,0,0)"

    # 5. Partikeleffekte & Aerosol-Fluten
    extra_fx = ""
    if s['active']:
        extra_fx += f"""
        <div style="position: absolute; left: 8px; bottom: 35px; width: 6px; height: 4px; background: {m['color']}; border-radius:2px; animation: chip_spray_left 0.1s infinite linear;"></div>
        <div style="position: absolute; right: 8px; bottom: 35px; width: 5px; height: 3px; background: {m['color']}; border-radius:1px; animation: chip_spray_right 0.09s infinite linear; animation-delay: 0.03s;"></div>
        """
        if kuehlung:
            extra_fx += """
            <div style="position: absolute; width: 3px; height: 75px; background: rgba(180, 240, 255, 0.7); filter: blur(0.5px); animation: kss_flood_left 0.12s infinite linear; transform-origin: top left;"></div>
            <div style="position: absolute; width: 3px; height: 75px; background: rgba(180, 240, 255, 0.7); filter: blur(0.5px); animation: kss_flood_right 0.12s infinite linear; transform-origin: top right;"></div>
            <div style="position: absolute; left: 50%; bottom: 32px; width: 30px; height: 15px; background: rgba(200, 235, 255, 0.15); filter: blur(4px); border-radius: 50%; animation: kss_mist 0.25s infinite ease-out;"></div>
            """
        if t_val > 220:
            extra_fx += '<div style="position: absolute; left: 50%; bottom: 35px; width: 18px; height: 18px; background: rgba(240,240,240,0.15); filter: blur(7px); border-radius: 50%; animation: smoke_rise 0.3s infinite linear;"></div>'
        if t_val > 420:
            extra_fx += """
            <div style="position: absolute; left: 12px; bottom: 35px; width: 3px; height: 3px; background: #ffcc00; box-shadow:0 0 5px #ff3300; border-radius:50%; animation: chip_spray_left 0.05s infinite linear;"></div>
            <div style="position: absolute; right: 12px; bottom: 35px; width: 3px; height: 3px; background: #fffa00; box-shadow:0 0 5px #ff3300; border-radius:50%; animation: chip_spray_right 0.05s infinite linear;"></div>
            """

    # 6. Kinetik-Zuweisung
    if s['broken']:
        anim_spin, anim_shake, anim_feed = "none", "none", "none"
        drill_render = """
        <div style="width: 44px; height: 60px; background: #2f2f2f; transform: translate(16px, -3px) rotate(-32deg); border-bottom: 4px dashed #ff2222; box-shadow: inset 4px 0 10px rgba(0,0,0,0.7);"></div>
        <div style="width: 44px; height: 50px; background: #111; transform: translate(-20px, 18px) rotate(60deg); clip-path: polygon(0% 0%, 100% 0%, 50% 100%);"></div>
        """
        status_label = "<span style='color:#ff7b72; font-weight:900;'>CRASH / BRUCH</span>"
    else:
        spin_duration = f"{max(0.010, 45.0 / (s['drehzahl'] + 1)):.3f}s" if s['active'] else "0s"
        anim_spin = f"helical_spin {spin_duration} linear infinite" if s['active'] else "none"
        shake_duration = f"{max(0.005, 0.06 / (s['vibration'] + 0.01)):.3f}s"
        anim_shake = f"industrial_shake {shake_duration} infinite linear" if s['active'] or s['stall'] else "none"
        anim_feed = "tool_feed 2.5s infinite ease-in-out" if s['active'] else "none"
        
        drill_render = f"""
        <div style="animation: {anim_spin}; width: 44px; height: 115px; 
                    background: linear-gradient(90deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 20%, rgba(0,0,0,0.4) 70%, rgba(0,0,0,0.7) 100%),
                                linear-gradient(to right, transparent 40%, rgba(255,255,255,0.15) 50%, transparent 60%),
                                repeating-linear-gradient(135deg, #12161b 0px, #12161b 12px, #3a3a3a 16px, #777 22px, #3a3a3a 26px, #12161b 38px); 
                    background-size: 100% 100%, 100% 100%, 100% 45px; box-shadow: inset 4px 0 10px rgba(0,0,0,0.75);"></div>
        <div style="background: {tip_color}; box-shadow: {glow_effect}; width: 44px; height: 21px; 
                    clip-path: polygon(0% 0%, 100% 0%, 50% 100%); margin-top: -1px;"></div>
        """
        status_label = "<span style='color:#2ea44f; font-weight:900;'>ROTATION LIVE</span>" if s['active'] else "<span style='color:#8b949e;'>STANDBY</span>"
        if s['stall']: status_label = "<span style='color:#e3b341; font-weight:900;'>STALL / BLOCKIERT</span>"

    st.html(f"""
        <div class="glass-card" style="padding: 20px; height: 100%; min-height: 330px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; background: #07090e; position: relative; overflow: hidden; border: 1px solid #21262d;">
            <span class="val-title" style="color: #bc8cff; font-size:1.05rem;">Realtime-VFX-Spindel</span>
            
            <div style="position: relative; display: flex; flex-direction: column; align-items: center; z-index:4;">
                <div style="width: 76px; height: 30px; background: linear-gradient(90deg, #161b22 0%, #485260 50%, #161b22 100%); border-radius: 4px 4px 0 0; border: 1px solid #30363d;"></div>
                <div style="{led_style} width: 66px; height: 6px; margin-top: -1px; border-radius: 0 0 2px 2px; transition: all 0.3s;"></div>
            </div>
            
            <div style="animation: {anim_feed}; width: 100%; display: flex; flex-direction: column; align-items: center; z-index:2;">
                <div style="animation: {anim_shake}; display: flex; flex-direction: column; align-items: center; position: relative;">
                    {drill_render}
                    {extra_fx}
                </div>
            </div>
            
            <div style="width: 115%; height: 26px; background: linear-gradient(180deg, {m['color']} 0%, #080a0f 100%); border-top: 2px solid rgba(255,255,255,0.15); border-radius: 2px; z-index: 3; margin-top:-2px; display:flex; justify-content:center; position: relative;">
                <div style="position: absolute; top: -3px; width: 22px; height: 5px; background: {surface_glow}; filter: blur(2px); border-radius: 50%; box-shadow: 0 0 8px {surface_glow}; transition: background 0.2s;"></div>
                <div style="width: 16px; height: 7px; background: rgba(0,0,0,0.7); clip-path: polygon(0% 0%, 100% 0%, 50% 100%); z-index: 5;"></div>
            </div>
            
            <div style="margin-top: 8px; font-size: 1.05rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.8px; background: #12161f; padding: 4px 16px; border-radius: 20px; border: 1px solid #21262d; text-align:center; min-width:150px;">{status_label}</div>
        </div>
    """)

# --- 7. UX-DASHBOARD-KPI PANELS ---
with col_metrics:
    c_cyc, c0, c1, c2, c3, c4 = st.columns(6)
    c_cyc.html(f'<div class="glass-card" style="border: 1px solid #bc8cff; background: rgba(188,140,255,0.03);"><span class="val-title" style="color:#bc8cff;">Bohrzyklen</span><br><span class="val-main" style="color:#bc8cff;">{s["zyklen_anzahl"]}</span></div>')
    c0.html(f'<div class="glass-card"><span class="val-title">Schnittzeit tc</span><br><span class="val-main" style="color:#58a6ff">{s["zyklus"]:.1f} <span style="font-size:16px">s</span></span></div>')
    
    i_color = "#2ea44f" if s['integritaet'] > 50 else ("#e3b341" if s['integritaet'] > 20 else "#f85149")
    c1.html(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:{i_color}">{s["integritaet"]:.1f}%</span></div>')
    
    r_color = "#2ea44f" if s['risk'] < 0.3 else ("#e3b341" if s['risk'] < 0.7 else "#f85149")
    c2.html(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:{r_color}">{s["risk"]:.1%}</span></div>')
    
    t_color = "#58a6ff" if s['thermik'] < 200 else ("#e3b341" if s['thermik'] < m['t_crit'] else "#f85149")
    c3.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:{t_color}">{s["thermik"]:.0f}°C</span></div>')
    
    v_color = "#58a6ff" if s['vibration'] < 2.5 else ("#e3b341" if s['vibration'] < 5.5 else "#f85149")
    c4.html(f'<div class="glass-card"><span class="val-title">Schwingung</span><br><span class="val-main" style="color:{v_color}">{s["vibration"]:.2f} <span style="font-size:14px">mm/s</span></span></div>')

# --- 8. TABS: LIVE TRENDS VS SCENARIO LAB ---
t1, t2 = st.tabs(["📈 Echtzeit-Telemetrie & Oszilloskop", "🔬 Prädiktives Was-Wäre-Wenn Simulationslabor"])

with t1:
    col_graph, col_log = st.columns([2.1, 1])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Strukturelle Werkzeugintegrität (%)", "Kinetische Lastprofile: Drehmoment (Nm) & Axialkraft (N)", "Prozessdynamik: Schwingung (mm/s) & Schnitttemperatur (°C)"))
            
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#2ea44f', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['m'], line=dict(color='#bc8cff', width=2.5), name="Moment"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['f']/100.0, line=dict(color='#1f6feb', width=2, dash='dash'), name="Axialkraft/100"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#a371f7', width=2.5), name="Vibration"), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#ff7b72', width=2.5), name="Temp"), 3, 1)
            
            fig.update_layout(height=580, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Simulation starten, um Telemetriedaten aufzuzeichnen.")
            
    with col_log:
        st.markdown("### 👁️ Erklärbare KI-Ursachendiagnose (XAI)")
        if s['logs']:
            html_str = '<div class="xai-container">'
            for l in s['logs'][:10]:
                bars = "".join([f'<div class="xai-feature-row"><span>{e[0]}</span><span>{e[1]:.1f}%</span></div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{e[1]}%"></div></div>' for e in l['evidenz'][:3]])
                html_str += f"""
                <div class="xai-card">
                    <div style="display:flex; justify-content:between; align-items:center; margin-bottom:6px;">
                        <span class="diag-badge">{l['info']['diag']}</span>
                        <span style="font-size:11px; color:#8b949e; margin-left:auto; font-family:monospace;">{l['zeit']}</span>
                    </div>
                    <div class="reason-text">{l['info']['exp']}</div>
                    <div class="sensor-snapshot">{l['info']['snapshot']}</div>
                    <div style="margin-top:8px;">{bars}</div>
                    <div class="action-text">{l['info']['act']}</div>
                </div>"""
            html_str += '</div>'
            st.html(html_str)

# --- 9. WAS-WÄRE-WENN LABOR ---
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
        l_kss_fac = 4.5 if lab_kss else 0.85  
        l_temp = (22.0 + (l_p_friction * 0.20) / l_kss_fac) * sensor_temp_gain
        
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
            lab_risk = l_combined_score * 0.18
        else:
            lab_risk = 0.15 + (l_combined_score - 0.75) * 2.3
        lab_risk = np.clip(lab_risk, 0.01, 0.99)
        
        lc1, lc2 = st.columns(2)
        lc1.html(f'<div class="glass-card"><span class="val-title">Errechnetes Drehmoment</span><br><span class="val-main" style="color:#bc8cff">{l_torque:.1f} Nm</span><br><span style="font-size:11px;color:#8b949e;font-weight:bold;">Bruch-Limit: {c_t:.1f} Nm</span></div>')
        lc2.html(f'<div class="glass-card"><span class="val-title">Errechnete Temperatur</span><br><span class="val-main" style="color:#ff7b72">{l_temp:.0f} °C</span><br><span style="font-size:11px;color:#8b949e;font-weight:bold;">Werkstoff-Erweichungs-Limit: {lm["t_crit"]} °C</span></div>')
        
        lc3, lc4 = st.columns(2)
        lc3.html(f'<div class="glass-card"><span class="val-title">Erwartete Spindellast</span><br><span class="val-main" style="color:#58a6ff">{l_power:.2f} kW</span><br><span style="font-size:11px;color:#8b949e;font-weight:bold;">Max. Motorleistung: 7.5 kW</span></div>')
        lc4.html(f'<div class="glass-card"><span class="val-title">Errechnete Vorschubkraft</span><br><span class="val-main" style="color:#1f6feb">{l_force:.0f} N</span><br><span style="font-size:11px;color:#8b949e;font-weight:bold;">Knick-Limit: {c_f:.0f} N</span></div>')
        
        st.html(f"""
            <div class="glass-card" style="border: 1px solid #58a6ff; margin-top:15px; background: rgba(88, 166, 255, 0.05); box-shadow: 0 0 20px rgba(88,166,255,0.15);">
                <span class="val-title" style="font-size:1.15rem; color:#58a6ff;">Präzisiertes KI-Bruchrisiko:</span><br>
                <span class="val-main" style="color:#ff7b72; font-size:3.2rem; margin-top:5px;">{lab_risk:.1%}</span>
            </div>
        """)
        
        st.markdown("### 🔬 Prädizierte KI-Ursachengewichtung:")
        html_bars = ""
        for name, prob in l_evidenz[:4]:
            html_bars += f"""
            <div class="xai-feature-row" style="margin-top:10px;"><span>{name}</span><span style="font-weight:bold; color:#58a6ff;">{prob:.1f}%</span></div>
            <div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{prob}%;"></div></div>
            """
        st.html(html_bars)

# --- 10. RUNTIME CONTROLS ---
st.divider()
b1, b2 = st.columns(2)
if b1.button("▶ SIMULATION STARTEN / PAUSIEREN", use_container_width=True):
    st.session_state.twin['active'] = not st.session_state.twin['active']
if b2.button("🔄 NEUES WERKZEUG EINSPANNEN (RESET)", use_container_width=True):
    st.session_state.twin = {
        'zyklus': 0.0, 'zyklen_anzahl': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }
    st.rerun()

if s['active']:
    if taktzeit > 0: time.sleep(taktzeit / 1000)
    st.rerun()
