import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-END INDUSTRIAL STYLING ---
st.set_page_config(layout="wide", page_title="KI-Zerspanungslabor TwinPro V4.5", page_icon="⚙️")

st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;800&display=swap');
    
    /* Globaler Next-Gen Darkmode */
    .stApp { 
        background-color: #080b10; 
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
        background: rgba(22, 27, 34, 0.7); 
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px; padding: 20px; margin-bottom: 15px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.4);
        box-shadow: 0 4px 25px rgba(88, 166, 255, 0.1);
    }
    .val-title { font-size: 1.05rem; color: #8b949e; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 800; color: #f0f6fc; margin-top: 5px; display: block; }
    
    /* XAI Diagnose-Karten */
    .xai-container { height: 600px; overflow-y: auto; padding-right: 5px; }
    .xai-card {
        background: #11151c; border-left: 6px solid #58a6ff;
        padding: 18px; border-radius: 8px; margin-bottom: 12px;
        border-top: 1px solid rgba(255,255,255,0.03);
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

    /* --- ULTIMATE HIGH-REALISM ANIMATION KEYFRAMES --- */
    @keyframes helical_spin {
        0% { background-position: 0px 0px, 0px 0px; }
        100% { background-position: 0px -160px, 0px 0px; }
    }
    @keyframes tool_feed {
        0% { transform: translateY(-6px); }
        50% { transform: translateY(12px); }
        100% { transform: translateY(-6px); }
    }
    @keyframes industrial_shake {
        0% { transform: translate(0.3px, 0.3px) rotate(0.03deg); }
        50% { transform: translate(-0.6px, 0.1px) rotate(-0.03deg); }
        100% { transform: translate(0.3px, -0.4px) rotate(0.02deg); }
    }
    @keyframes chip_spray_left {
        0% { transform: translate(0, 0) scale(1) rotate(0deg); opacity: 1; }
        80% { opacity: 0.8; }
        100% { transform: translate(-70px, -40px) scale(0.1) rotate(-360deg); opacity: 0; }
    }
    @keyframes chip_spray_right {
        0% { transform: translate(0, 0) scale(1) rotate(0deg); opacity: 1; }
        80% { opacity: 0.8; }
        100% { transform: translate(70px, -40px) scale(0.1) rotate(360deg); opacity: 0; }
    }
    @keyframes smoke_rise {
        0% { transform: translate(-50%, -5px) scale(0.6); opacity: 0; }
        40% { opacity: 0.3; }
        100% { transform: translate(-50%, -60px) scale(3.0); opacity: 0; }
    }
    @keyframes kss_flow_left {
        0% { transform: translate(-40px, -60px) scaleX(1) rotate(45deg); opacity: 0.8; }
        100% { transform: translate(-4px, -5px) scaleX(0.2) rotate(45deg); opacity: 1; }
    }
    @keyframes kss_flow_right {
        0% { transform: translate(40px, -60px) scaleX(1) rotate(-45deg); opacity: 0.8; }
        100% { transform: translate(4px, -5px) scaleX(0.2) rotate(-445deg); opacity: 1; }
    }
    </style>
""")

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
        'zyklus': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
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

# --- 5. PHYSIK-ENGINE (REAL-TIME ADAPTED) ---
n = (vc * 1000) / (np.pi * d) if d > 0 else 0
s['drehzahl'] = n

if s['active'] and not s['broken'] and not s['stall']:
    dt = (taktzeit / 1000.0 if taktzeit > 0 else 0.05) * schrittweite
    s['zyklus'] += dt # Akkumuliert reale Schnittsekunden
    
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

# --- 6. VISUELLES HIGH-FIDELITY UPGRADE ---
if s['broken']:
    st.html('<div class="emergency-alert">💥 STRUKTURELLER WERKZEUGBRUCH! Schaft durch mechanische Überlast komplett zerstört.</div>')
if s['stall']:
    st.html('<div class="emergency-alert">⚠️ MOTOR-STALL: Leistungsaufnahme überschreitet maximales Drehmoment der Spindel (7.5 kW).</div>')

col_animation, col_metrics = st.columns([1.4, 4])

with col_animation:
    t_val = s['thermik']
    integ_val = s['integritaet']
    
    # Thermischer Gradient & Verschleiß-Überlagerung an der Spitze
    if t_val < 150:
        tip_base = "#555555"
        glow_effect = "rgba(0,0,0,0)"
    elif t_val < 380:
        factor = (t_val - 150) / 230
        r = int(85 + (165 - 85) * factor)
        g = int(85 + (65 - 85) * factor)
        b = int(85 + (20 - 85) * factor)
        tip_base = f"rgb({r},{g},{b})"
        glow_effect = f"0 8px 20px rgba(255, 90, 0, {0.25 * factor})"
    else:
        factor = min(1.0, (t_val - 380) / 320)
        r = int(165 + (90 * factor))
        g = int(65 * (1.0 - factor))
        tip_base = f"rgb({r}, {g}, 0)"
        glow_effect = f"0 8px 25px rgba({r}, {g}, 0, {0.4 + 0.5 * factor})"

    # Mischung aus Hitze und mechanischer Graufärbung/Abnutzung
    wear_factor = (100.0 - integ_val) / 100.0
    if not s['broken']:
        # Blende progressiv Brandspuren/Verschleiß ein
        tip_color = f"linear-gradient(to bottom, #444 0%, {tip_base} 70%, rgba(20,20,20,{wear_factor:.2f}) 100%)"
    else:
        tip_color = "#222"

    # Partikel- und KSS-Injektionen
    extra_fx = ""
    if s['active']:
        # Metallspäne-Flug
        extra_fx += f"""
        <div style="position: absolute; left: 8px; bottom: 35px; width: 6px; height: 4px; background: {m['color']}; border-radius:2px; animation: chip_spray_left 0.12s infinite linear;"></div>
        <div style="position: absolute; right: 8px; bottom: 35px; width: 5px; height: 3px; background: {m['color']}; border-radius:1px; animation: chip_spray_right 0.10s infinite linear; animation-delay: 0.04s;"></div>
        """
        # Kühlmittelstrahlen (falls aktiv)
        if kuehlung:
            extra_fx += """
            <div style="position: absolute; width: 3px; height: 70px; background: rgba(180, 230, 255, 0.6); border-radius: 2px; filter: blur(1px); animation: kss_flow_left 0.15s infinite linear;"></div>
            <div style="position: absolute; width: 3px; height: 70px; background: rgba(180, 230, 255, 0.6); border-radius: 2px; filter: blur(1px); animation: kss_flow_right 0.15s infinite linear;"></div>
            """
        # Qualm & glühende Funken bei Hitze
        if t_val > 250:
            extra_fx += """
            <div style="position: absolute; left: 50%; bottom: 35px; width: 16px; height: 16px; background: rgba(230,230,230,0.12); filter: blur(6px); border-radius: 50%; animation: smoke_rise 0.35s infinite linear;"></div>
            """
        if t_val > 400:
            extra_fx += """
            <div style="position: absolute; left: 15px; bottom: 35px; width: 3px; height: 3px; background: #ffaa00; box-shadow:0 0 4px #ff4400; border-radius:50%; animation: chip_spray_left 0.07s infinite linear;"></div>
            <div style="position: absolute; right: 15px; bottom: 35px; width: 3px; height: 3px; background: #ffcc00; box-shadow:0 0 4px #ff4400; border-radius:50%; animation: chip_spray_right 0.06s infinite linear;"></div>
            """

    # Dynamische Zuweisung von Drehzahl-Frequenz & Vibrationen
    if s['broken']:
        anim_spin, anim_shake, anim_feed = "none", "none", "none"
        drill_render = """
        <div style="width: 44px; height: 60px; background: #3a3a3a; transform: translate(14px, -5px) rotate(-30deg); border-bottom: 3px dashed #ff3333; box-shadow: inset 3px 0 10px rgba(0,0,0,0.6);"></div>
        <div style="width: 44px; height: 50px; background: #1a1a1a; transform: translate(-18px, 20px) rotate(55deg); clip-path: polygon(0% 0%, 100% 0%, 50% 100%); border: 1px solid #333;"></div>
        """
        status_label = "<span style='color:#ff7b72; font-weight:900;'>CRASH / BRUCH</span>"
    else:
        spin_duration = f"{max(0.012, 50.0 / (s['drehzahl'] + 1)):.3f}s" if s['active'] else "0s"
        anim_spin = f"helical_spin {spin_duration} linear infinite" if s['active'] else "none"
        shake_duration = f"{max(0.006, 0.07 / (s['vibration'] + 0.01)):.3f}s"
        anim_shake = f"industrial_shake {shake_duration} infinite linear" if s['active'] or s['stall'] else "none"
        anim_feed = "tool_feed 2.5s infinite ease-in-out" if s['active'] else "none"
        
        drill_render = f"""
        <div style="animation: {anim_spin}; width: 44px; height: 115px; 
                    background: linear-gradient(to right, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 25%, rgba(0,0,0,0.5) 75%, rgba(0,0,0,0.8) 100%),
                                repeating-linear-gradient(135deg, #15191e 0px, #15191e 12px, #444 16px, #888 22px, #444 26px, #15191e 38px); 
                    background-size: 100% 100%, 100% 45px; box-shadow: inset 4px 0 10px rgba(0,0,0,0.7); border-radius: 0 0 1px 1px;"></div>
        <div style="background: {tip_color}; box-shadow: {glow_effect}; width: 44px; height: 21px; 
                    clip-path: polygon(0% 0%, 100% 0%, 50% 100%); margin-top: -1px; transition: background 0.15s;"></div>
        """
        status_label = "<span style='color:#2ea44f; font-weight:900;'>ROTATION LIVE</span>" if s['active'] else "<span style='color:#8b949e;'>STANDBY</span>"
        if s['stall']: status_label = "<span style='color:#e3b341; font-weight:900;'>STALL / BLOCKIERT</span>"

    st.html(f"""
        <div class="glass-card" style="padding: 20px; height: 100%; min-height: 310px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; background: #0b0e14; position: relative; overflow: hidden;">
            <span class="val-title" style="color: #58a6ff; font-size:1.05rem;">Spindel-Zerspanungsraum</span>
            
            <div style="width: 75px; height: 32px; background: linear-gradient(90deg, #1c212b 0%, #5c6673 50%, #1c212b 100%); border-radius: 4px 4px 0 0; border: 1px solid #30363d; box-shadow: 0 3px 8px rgba(0,0,0,0.5); z-index:2;"></div>
            
            <div style="animation: {anim_feed}; width: 100%; display: flex; flex-direction: column; align-items: center; z-index:1;">
                <div style="animation: {anim_shake}; display: flex; flex-direction: column; align-items: center; position: relative;">
                    {drill_render}
                    {extra_fx}
                </div>
            </div>
            
            <div style="width: 115%; height: 25px; background: linear-gradient(180deg, {m['color']} 0%, #0d1117 100%); border-top: 2px solid rgba(255,255,255,0.15); border-radius: 3px; z-index: 3; margin-top:-2px; display:flex; justify-content:center;">
                <div style="width: 16px; height: 8px; background: rgba(0,0,0,0.6); clip-path: polygon(0% 0%, 100% 0%, 50% 100%);"></div>
            </div>
            
            <div style="margin-top: 10px; font-size: 1.05rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.8px; background: #161b22; padding: 5px 16px; border-radius: 20px; border: 1px solid #30363d; text-align:center; min-width:150px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);">{status_label}</div>
        </div>
    """)

# --- 7. UX-DASHBOARD-KPI PANELS ---
with col_metrics:
    c0, c1, c2, c3, c4, c5 = st.columns(6)
    c0.html(f'<div class="glass-card"><span class="val-title">Schnittzeit tc</span><br><span class="val-main" style="color:#58a6ff">{s["zyklus"]:.1f} <span style="font-size:16px">s</span></span></div>')
    
    i_color = "#2ea44f" if s['integritaet'] > 50 else ("#e3b341" if s['integritaet'] > 20 else "#f85149")
    c1.html(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:{i_color}">{s["integritaet"]:.1f}%</span></div>')
    
    r_color = "#2ea44f" if s['risk'] < 0.3 else ("#e3b341" if s['risk'] < 0.7 else "#f85149")
    c2.html(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:{r_color}">{s["risk"]:.1%}</span></div>')
    
    t_color = "#58a6ff" if s['thermik'] < 200 else ("#e3b341" if s['thermik'] < m['t_crit'] else "#f85149")
    c3.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:{t_color}">{s["thermik"]:.0f}°C</span></div>')
    
    v_color = "#58a6ff" if s['vibration'] < 2.5 else ("#e3b341" if s['vibration'] < 5.5 else "#f85149")
    c4.html(f'<div class="glass-card"><span class="val-title">Schwingung</span><br><span class="val-main" style="color:{v_color}">{s["vibration"]:.2f} <span style="font-size:14px">mm/s</span></span></div>')
    
    c5.html(f'<div class="glass-card"><span class="val-title">Drehmoment</span><br><span class="val-main" style="color:#bc8cff">{s["drehmoment"]:.1f} <span style="font-size:16px">Nm</span></span></div>')

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
        'zyklus': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }
    st.rerun()

if s['active']:
    if taktzeit > 0: time.sleep(taktzeit / 1000)
    st.rerun()
