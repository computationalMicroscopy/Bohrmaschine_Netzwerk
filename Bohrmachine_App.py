import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-END INDUSTRIAL STYLING ---
st.set_page_config(layout="wide", page_title="TwinPro V5.6 | Professional AI Lab", page_icon="⚙️")

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
    
    vc = st.slider("Live Schnittgeschw. vc [m/min]", 30, 350, 100)
    f = st.slider("Live Vorschub f [mm/U]", 0.05, 0.60, 0.15)
    d = st.slider("Live Bohrer-Durchmesser d [mm]", 5.0, 32.0, 12.0)
    kuehlung = st.toggle("KSS Kühlung Aktiv", value=True)
    
    st.divider()
    st.header("🎛️ Zeitskalierung & Sensoren")
    schrittweite = st.number_input("Zeitskalierungsfaktor", min_value=1, max_value=1000000, value=5)
    taktzeit = st.select_slider("Aktualisierungsintervall (ms)", options=[500, 200, 100, 0], value=100)
    
    sensor_temp_gain = st.slider("Sensor-Empfindlichkeit (T)", 0.5, 2.5, 1.0, step=0.1)
    sensor_vibr_gain = st.slider("Sensor-Empfindlichkeit (V)", 0.5, 3.0, 1.0, step=0.1)
    noise_level = st.slider("Rausch-Amplitude (Vibration)", 0.1, 2.0, 0.5)

# DYNAMISCHE ANIMATIONS-ZEITBERECHNUNG
basis_zyklus_zeit = 2.5 
dynamische_animations_dauer = max(0.05, basis_zyklus_zeit / schrittweite)

# --- GLOBAL DYNAMIC CSS STYLES ---
st.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;800&display=swap');
    
    .stApp {{ background-color: #06090e; color: #c9d1d9; font-family: 'Inter', sans-serif; }}
    
    .main-title {{
        font-size: 2.8rem; font-weight: 800; color: #f0f6fc;
        text-align: center; margin-bottom: 30px; padding-bottom: 20px;
        border-bottom: 1px solid rgba(240, 246, 252, 0.1);
        letter-spacing: -0.5px; background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    
    .glass-card {{
        background: rgba(16, 22, 30, 0.85); border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 12px; padding: 20px; margin-bottom: 15px; text-align: center;
        box-shadow: 0 4px 25px rgba(0,0,0,0.3); transition: all 0.3s ease;
    }}
    
    .val-title {{ font-size: 1.05rem; color: #8b949e; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }}
    .val-main {{ font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 800; color: #f0f6fc; margin-top: 5px; display: block; }}
    
    .xai-container {{ height: 500px; overflow-y: auto; padding-right: 5px; }}
    .xai-card {{ background: #0d1117; border-left: 6px solid #58a6ff; padding: 18px; border-radius: 8px; margin-bottom: 12px; border-top: 1px solid rgba(255,255,255,0.02); }}
    .xai-feature-row {{ display: flex; justify-content: space-between; font-size: 1.1rem; color: #c9d1d9; margin-top: 6px; font-weight: 500;}}
    .xai-bar-bg {{ background: #21262d; height: 8px; width: 100%; border-radius: 4px; margin-bottom: 8px; overflow:hidden;}}
    .xai-bar-fill {{ background: linear-gradient(90deg, #58a6ff, #1f6feb); height: 100%; border-radius: 4px; }}
    .reason-text {{ color: #f0f6fc; font-size: 1.25rem; margin-top: 6px; font-weight: 700; }}
    .sensor-snapshot {{ font-size: 1.0rem; color: #8b949e; margin-top: 6px; font-family: 'JetBrains Mono', monospace; border-bottom: 1px solid #30363d; padding-bottom: 6px;}}
    .action-text {{ color: #ff7b72; font-weight: bold; font-size: 1.15rem; margin-top: 8px; border-top: 1px solid rgba(48, 54, 61, 0.5); padding-top: 8px; }}
    .diag-badge {{ background: #23426f; color: #58a6ff; border: 1px solid #388bfd; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;}}

    @keyframes helical_spin {{ 0% {{ background-position: 0px 0px, 0px 0px; }} 100% {{ background-position: 0px -180px, 0px 0px; }} }}
    @keyframes tool_feed_optimized {{ 0%, 15%, 90%, 100% {{ transform: translateY(-35px); }} 25% {{ transform: translateY(-2px); }} 65%, 75% {{ transform: translateY(22px); }} }}
    @keyframes vfx_sync_gate {{ 0%, 24%, 76%, 100% {{ opacity: 0; transform: scale(0.5); }} 25%, 65% {{ opacity: 1; transform: scale(1); }} }}
    @keyframes led_pulse {{ 0%, 100% {{ box-shadow: 0 0 10px var(--led-color), inset 0 0 5px var(--led-color); }} 50% {{ box-shadow: 0 0 26px var(--led-color), inset 0 0 12px var(--led-color); }} }}
    @keyframes strobe_crit {{ 0%, 100% {{ background: #ff0000; box-shadow: 0 0 25px #ff0000; }} 50% {{ background: #200000; box-shadow: 0 0 2px #200000; }} }}
    </style>
""")

st.html('<div class="main-title">🚀 TwinPro | Industrie-Zwillings-Dashboard</div>')

# --- 2. DETERMINISTISCHE REAL-SENSOR KI DIAGNOSE ENGINE ---
def compute_sensor_diagnostics(current_vals, settings, integrity, kuehlung, m):
    M, T, V, F = current_vals['M'], current_vals['T'], current_vals['V'], current_vals['F']
    vc, f, d = settings['vc'], settings['f'], settings['d']
    
    crit_torque = 0.12 * (d ** 3) 
    crit_force = 320 * (d ** 2)
    t_crit = m['t_crit']
    
    evidenz = {
        "Mechanische Torsions-Überlast": min(100.0, (M / crit_torque) * 100.0) if M > crit_torque * 0.85 else 0.0,
        "Thermische Gefüge-Erweichung": min(100.0, (T / t_crit) * 100.0) if T > t_crit * 0.85 else 0.0,
        "Regeneratives Rattern (Resonanz)": min(100.0, (V / 8.0) * 100.0) if V > 5.0 else 0.0,
        "Aufbauschneidenbildung (Adhäsion)": 85.0 + (f * 20.0) if vc < 60 and f > 0.12 and T < 200 else 0.0,
        "Axiale Schaft-Knickung": min(100.0, (F / crit_force) * 100.0) if F > crit_force * 0.85 else 0.0,
        "Normalbetrieb": 10.0
    }

    sorted_evidenz = sorted(evidenz.items(), key=lambda x: x[1], reverse=True)
    top_reason = sorted_evidenz[0][0]
    
    mapping = {
        "Mechanische Torsions-Überlast": {"diag": "CRITICAL TORQUE", "exp": f"Das gemessene Drehmoment von {M:.1f} Nm übersteigt die Scherspannungsgrenze.", "act": f"MINDERN SIE DEN VORSCHUB auf {f*0.7:.2f} mm/U."},
        "Thermische Gefüge-Erweichung": {"diag": "THERMAL OVERLOAD", "exp": f"Die Schnittkantentemperatur ({T:.0f}°C) hat die Anlasstemperatur überschritten.", "act": f"Schnittgeschwindigkeit vc auf {int(vc*0.75)} m/min drosseln."},
        "Regeneratives Rattern (Resonanz)": {"diag": "RESONANT CHATTER", "exp": f"Selbsterregte Schwingungen ({V:.1f} mm/s) stören den Prozess.", "act": f"Ändern Sie vc um +15% auf {int(vc*1.15)} m/min."},
        "Aufbauschneidenbildung (Adhäsion)": {"diag": "ADHESION DETECTED", "exp": "Die Schnittgeschwindigkeit ist zu gering. Kaltverschweißung droht.", "act": f"Fahren Sie vc auf ({int(vc*1.2)} m/min) hoch."},
        "Axiale Schaft-Knickung": {"diag": "AXIAL BUCKLED", "exp": f"Axiale Vorschubkraft ({F:.0f} N) nähert sich der kritischen Knicklast.", "act": f"Reduzieren Sie f umgehend auf {f*0.5:.2f} mm/U."},
        "Normalbetrieb": {"diag": "NOMINAL", "exp": "System läuft im stabilen Bereich.", "act": "Keine Korrekturdaten nötig."}
    }
    
    res = mapping.get(top_reason, mapping["Normalbetrieb"])
    res["snapshot"] = f"M: {M:.1f}Nm | T: {T:.0f}°C | V: {V:.1f}mm/s | F_f: {F:.0f}N"
    return res, sorted_evidenz

# --- 3. PHYSIK-ENGINE KINETIK & THERMODYNAMIK ---
n = (vc * 1000) / (np.pi * d) if d > 0 else 0
s['drehzahl'] = n

if s['active'] and not s['broken'] and not s['stall']:
    dt = (taktzeit / 1000.0 if taktzeit > 0 else 0.05) * schrittweite
    
    altes_zeitfenster = int(s['zyklus'] / 2.5)
    s['zyklus'] += dt 
    neues_zeitfenster = int(s['zyklus'] / 2.5)
    
    if neues_zeitfenster > altes_zeitfenster:
        s['zyklen_anzahl'] += (neues_zeitfenster - altes_zeitfenster)
    
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
    norm_wear = (100.0 - s['integritaet']) / 100.0
    
    exp_report, evidenz_list = compute_sensor_diagnostics(
        {'M': s['drehmoment'], 'T': s['thermik'], 'V': s['vibration'], 'F': s['vorschubkraft']},
        {'vc': vc, 'f': f, 'd': d}, s['integritaet'], kuehlung, m
    )
    
    combined_risk_score = max([norm_torque, norm_force, norm_temp, s['vibration']/8.0]) + (norm_wear ** 2.0) * 0.95
    s['risk'] = np.clip(combined_risk_score * 0.18 if combined_risk_score < 0.75 else 0.15 + (combined_risk_score - 0.75) * 2.3, 0.01, 0.99)
    
    v_factor = (vc / 120.0) ** 1.5
    thermal_accelerator = np.exp(max(0.0, s['thermik'] - m['t_crit']) / 30.0)
    s['abrasion'] += (m['wear_factor'] * v_factor * f * thermal_accelerator) * dt
    
    fatigue = 0.0
    if norm_torque > 0.85: fatigue += (norm_torque - 0.85) ** 2
    if norm_force > 0.85: fatigue += (norm_force - 0.85) ** 2
    
    total_wear_increment = ((s['abrasion'] * 0.02) + fatigue * 0.5) * dt * 10.0
    s['integritaet'] = max(0.0, s['integritaet'] - total_wear_increment)
    
    if s['leistung'] > 7.5:
        s['stall'] = True; s['active'] = False
    
    if s['integritaet'] <= 0.0 or norm_torque > 1.15 or norm_force > 1.2 or (norm_temp > 1.1 and s['drehmoment'] > crit_torque * 0.5):
        s['broken'] = True; s['active'] = False; s['integritaet'] = 0.0
    
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'info': exp_report, 'evidenz': evidenz_list})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'm': s['drehmoment'], 't': s['thermik'], 'v': s['vibration'], 'f': s['vorschubkraft']})

# --- 4. DYNAMISCHE GEOMETRIE-SKALIERUNG & VFX ---
scale_factor = d / 12.0
drill_width_px = int(44 * scale_factor)
drill_tip_height_px = int(21 * scale_factor)
shake_offset_px = max(1.0, 1.5 * s['vibration'] * scale_factor)
spray_left_target_x = int(-90 * scale_factor)
spray_right_target_x = int(90 * scale_factor)
spray_y_target = int(-45 * scale_factor)
kss_flood_start_x = int(50 * scale_factor)

st.html(f"""
    <style>
    @keyframes industrial_shake {{
        0%, 100% {{ transform: translate(0, 0); }}
        20% {{ transform: translate(-{shake_offset_px:.1f}px, {shake_offset_px/2:.1f}px); }}
        40% {{ transform: translate({shake_offset_px:.1f}px, -{shake_offset_px:.1f}px); }}
        60% {{ transform: translate(-{shake_offset_px/2:.1f}px, -{shake_offset_px/2:.1f}px); }}
        80% {{ transform: translate({shake_offset_px:.1f}px, {shake_offset_px:.1f}px); }}
    }}
    @keyframes chip_spray_left {{
        0% {{ transform: translate(0, 0) scale(1.3) rotate(0deg); opacity: 1; }}
        100% {{ transform: translate({spray_left_target_x}px, {spray_y_target}px) scale(0.1) rotate(-720deg); opacity: 0; }}
    }}
    @keyframes chip_spray_right {{
        0% {{ transform: translate(0, 0) scale(1.3) rotate(0deg); opacity: 1; }}
        100% {{ transform: translate({spray_right_target_x}px, {spray_y_target}px) scale(0.1) rotate(720deg); opacity: 0; }}
    }}
    @keyframes kss_flood_left {{
        0% {{ transform: translate(-{kss_flood_start_x}px, -70px) scaleY(1) rotate(42deg); opacity: 0.6; }}
        100% {{ transform: translate(-3px, -2px) scaleX(0.3) rotate(42deg); opacity: 1; }}
    }}
    @keyframes kss_flood_right {{
        0% {{ transform: translate({kss_flood_start_x}px, -70px) scaleY(1) rotate(-42deg); opacity: 0.6; }}
        100% {{ transform: translate(3px, -2px) scaleX(0.3) rotate(-42deg); opacity: 1; }}
    }}
    </style>
""")

col_vfx, col_kpi = st.columns([1.5, 4])

with col_vfx:
    t_val = s['thermik']
    if s['broken'] or s['stall']:
        led_style = "animation: strobe_crit 0.2s infinite;"
    elif not s['active']:
        led_style = "background: #555; box-shadow: 0 0 5px #333; --led-color: #555;"
    else:
        if s['risk'] > 0.7: color, hex_val = "rgba(255, 68, 0, 1)", "#ff4400"
        elif kuehlung: color, hex_val = "rgba(0, 180, 255, 1)", "#00b4ff"
        else: color, hex_val = "rgba(46, 164, 79, 1)", "#2ea44f"
        led_style = f"background: {hex_val}; --led-color: {color}; animation: led_pulse 1s infinite ease-in-out;"

    if t_val < 150: tip_base, glow_effect = "#555555", "rgba(0,0,0,0)"
    elif t_val < 380:
        factor = (t_val - 150) / 230
        tip_base = f"rgb({int(85+80*factor)},{int(85-20*factor)},{int(85-65*factor)})"
        glow_effect = f"0 8px 20px rgba(255, 90, 0, {0.3 * factor})"
    else:
        factor = min(1.0, (t_val - 380) / 320)
        tip_base = f"rgb({int(165+90*factor)}, {int(65*(1.0-factor))}, 0)"
        glow_effect = f"0 8px 30px rgba({int(165+90*factor)}, {int(65*(1.0-factor))}, 0, {0.5 + 0.4 * factor})"

    wear_factor = (100.0 - s['integritaet']) / 100.0
    tip_color = f"linear-gradient(to bottom, #444 0%, {tip_base} 65%, rgba(15,15,15,{wear_factor:.2f}) 100%)" if not s['broken'] else "#1a1a1a"
    surface_glow = f"rgba(255, 60, 0, {min(0.9, (t_val-100)/500)})" if t_val > 100 else "rgba(0,0,0,0)"

    synchronized_vfx = ""
    if s['active']:
        synchronized_vfx += f"""
        <div style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; animation: vfx_sync_gate {dynamische_animations_dauer}s infinite ease-in-out; pointer-events: none;">
            <div style="position: absolute; left: {int(8*scale_factor)}px; bottom: 35px; width: {max(3, int(6*scale_factor))}px; height: 4px; background: {m['color']}; border-radius:2px; animation: chip_spray_left 0.1s infinite linear;"></div>
            <div style="position: absolute; right: {int(8*scale_factor)}px; bottom: 35px; width: {max(2, int(5*scale_factor))}px; height: 3px; background: {m['color']}; border-radius:1px; animation: chip_spray_right 0.09s infinite linear; animation-delay: 0.03s;"></div>
        """
        if kuehlung:
            synchronized_vfx += """
            <div style="position: absolute; width: 3px; height: 75px; background: rgba(180, 240, 255, 0.7); filter: blur(0.5px); animation: kss_flood_left 0.12s infinite linear; transform-origin: top left;"></div>
            <div style="position: absolute; width: 3px; height: 75px; background: rgba(180, 240, 255, 0.7); filter: blur(0.5px); animation: kss_flood_right 0.12s infinite linear; transform-origin: top right;"></div>
            """
        synchronized_vfx += "</div>"

    if s['broken']:
        anim_spin, anim_shake, anim_feed = "none", "none", "none"
        drill_render = f"""
        <div style="width: {drill_width_px}px; height: 60px; background: #2f2f2f; transform: translate(16px, -3px) rotate(-32deg); border-bottom: 4px dashed #ff2222; box-shadow: inset 4px 0 10px rgba(0,0,0,0.7);"></div>
        <div style="width: {drill_width_px}px; height: 50px; background: #111; transform: translate(-20px, 18px) rotate(60deg); clip-path: polygon(0% 0%, 100% 0%, 50% 100%);"></div>
        """
        status_label = "<span style='color:#ff7b72; font-weight:900;'>CRASH / BRUCH</span>"
    else:
        spin_duration = f"{max(0.010, 45.0 / (s['drehzahl'] + 1)):.3f}s" if s['active'] else "0s"
        anim_spin = f"helical_spin {spin_duration} linear infinite" if s['active'] else "none"
        shake_duration = f"{max(0.005, 0.06 / (s['vibration'] + 0.01)):.3f}s"
        anim_shake = f"industrial_shake {shake_duration} infinite linear" if s['active'] or s['stall'] else "none"
        anim_feed = f"tool_feed_optimized {dynamische_animations_dauer}s infinite ease-in-out" if s['active'] else "none"
        
        drill_render = f"""
        <div style="animation: {anim_spin}; width: {drill_width_px}px; height: 115px; 
                    background: linear-gradient(90deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 20%, rgba(0,0,0,0.4) 70%, rgba(0,0,0,0.7) 100%),
                                linear-gradient(to right, transparent 40%, rgba(255,255,255,0.15) 50%, transparent 60%),
                                repeating-linear-gradient(135deg, #12161b 0px, #12161b 12px, #3a3a3a 16px, #777 22px, #3a3a3a 26px, #12161b 38px); 
                    background-size: 100% 100%, 100% 100%, 100% 45px; box-shadow: inset 4px 0 10px rgba(0,0,0,0.75);"></div>
        <div style="background: {tip_color}; box-shadow: {glow_effect}; width: {drill_width_px}px; height: {drill_tip_height_px}px; clip-path: polygon(0% 0%, 100% 0%, 50% 100%); margin-top: -1px;"></div>
        """
        status_label = "<span style='color:#2ea44f; font-weight:900;'>ROTATION LIVE</span>" if s['active'] else "<span style='color:#8b949e;'>STANDBY</span>"
        if s['stall']: status_label = "<span style='color:#e3b341; font-weight:900;'>STALL / BLOCKIERT</span>"

    st.html(f"""
        <div class="glass-card" style="padding: 20px; height: 100%; min-height: 340px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; background: #07090e; position: relative; overflow: hidden; border: 1px solid #21262d;">
            <div style="position: relative; display: flex; flex-direction: column; align-items: center; z-index:4;">
                <div style="width: {int(76*scale_factor)}px; height: 30px; background: linear-gradient(90deg, #161b22 0%, #485260 50%, #161b22 100%); border-radius: 4px 4px 0 0; border: 1px solid #30363d;"></div>
                <div style="{led_style} width: {int(66*scale_factor)}px; height: 6px; margin-top: -1px; border-radius: 0 0 2px 2px; transition: all 0.3s;"></div>
            </div>
            <div style="animation: {anim_feed}; width: 100%; display: flex; flex-direction: column; align-items: center; z-index:2;">
                <div style="animation: {anim_shake}; display: flex; flex-direction: column; align-items: center; position: relative;">
                    {drill_render}
                    {synchronized_vfx}
                </div>
            </div>
            <div style="width: 115%; height: 26px; background: linear-gradient(180deg, {m['color']} 0%, #080a0f 100%); border-top: 2px solid rgba(255,255,255,0.15); border-radius: 2px; z-index: 3; margin-top:-2px; display:flex; justify-content:center; position: relative;">
                <div style="position: absolute; top: -3px; width: 22px; height: 5px; background: {surface_glow}; filter: blur(2px); border-radius: 50%; box-shadow: 0 0 8px {surface_glow}; transition: background 0.2s;"></div>
                <div style="width: 16px; height: 7px; background: rgba(0,0,0,0.7); clip-path: polygon(0% 0%, 100% 0%, 50% 100%); z-index: 5;"></div>
            </div>
            <div style="margin-top: 8px; font-size: 1.05rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.8px; background: #12161f; padding: 4px 16px; border-radius: 20px; border: 1px solid #21262d; text-align:center; min-width:150px;">{status_label}</div>
        </div>
    """)

# --- 5. UX-DASHBOARD-KPI PANELS ---
with col_kpi:
    c_cyc, c0, c1, c2, c3, c4 = st.columns(6)
    c_cyc.html(f'<div class="glass-card" style="border: 1px solid #bc8cff; background: rgba(188,140,255,0.03);"><span class="val-title" style="color:#bc8cff;">Bohrzyklen</span><br><span class="val-main" style="color:#bc8cff;">{s["zyklen_anzahl"]}</span></div>')
    c0.html(f'<div class="glass-card"><span class="val-title">Schnittzeit tc</span><br><span class="val-main" style="color:#58a6ff">{s["zyklus"]:.1f} s</span></div>')
    i_color = "#2ea44f" if s['integritaet'] > 50 else ("#e3b341" if s['integritaet'] > 20 else "#f85149")
    c1.html(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:{i_color}">{s["integritaet"]:.1f}%</span></div>')
    r_color = "#2ea44f" if s['risk'] < 0.3 else ("#e3b341" if s['risk'] < 0.7 else "#f85149")
    c2.html(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:{r_color}">{s["risk"]:.1%}</span></div>')
    t_color = "#58a6ff" if s['thermik'] < 200 else ("#e3b341" if s['thermik'] < m['t_crit'] else "#f85149")
    c3.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:{t_color}">{s["thermik"]:.0f}°C</span></div>')
    v_color = "#58a6ff" if s['vibration'] < 2.5 else ("#e3b341" if s['vibration'] < 5.5 else "#f85149")
    c4.html(f'<div class="glass-card"><span class="val-title">Schwingung</span><br><span class="val-main" style="color:{v_color}">{s["vibration"]:.2f}</span></div>')

# --- 6. TABS & TELEMETRIE-OSZILLOSKOP ---
t1, t2 = st.tabs(["📈 Echtzeit-Telemetrie & Oszilloskop", "🔬 Labor-Sandbox: Szenario-Analyse"])

with t1:
    col_graph, col_log = st.columns([2.1, 1])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Strukturelle Werkzeugintegrität (%)", "Kinetische Lastprofile (Nm & N)", "Prozessdynamik (mm/s & °C)"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', name="Integrität", line=dict(color='#2ea44f', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['m'], name="Drehmoment", line=dict(color='#bc8cff', width=2.5)), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['f'], name="Vorschubkraft", line=dict(color='#1f6feb', width=2, dash='dash')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], name="Schwingung", line=dict(color='#a371f7', width=2.5)), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], name="Temperatur", line=dict(color='#ff7b72', width=2)), 3, 1)
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=40, b=20), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Starten Sie die Maschine über den Button unten, um Live-Daten zu akquirieren.")

    with col_log:
        st.markdown('<span class="val-title">🤖 Explainable AI (XAI) Diagnostic Feed</span>', unsafe_allow_html=True)
        if s['logs']:
            st.markdown(f'<div class="xai-container">', unsafe_allow_html=True)
            for lg in s['logs'][:10]:
                st.markdown(f"""
                    <div class="xai-card">
                        <div style="display:flex; justify-content:between; align-items:center;">
                            <span class="diag-badge">{lg['info']['diag']}</span>
                            <span style="margin-left:auto; color:#8b949e; font-size:12px;">🕒 {lg['zeit']}</span>
                        </div>
                        <div class="reason-text">{lg['info']['exp']}</div>
                        <div class="sensor-snapshot">{lg['info']['snapshot']}</div>
                        <div class="action-text">💡 KI-Empfehlung: {lg['info']['act']}</div>
                        <div style="margin-top:10px; font-size:11px; color:#8b949e; font-weight:bold;">Neuronale Evidenz-Gewichtung:</div>
                """, unsafe_allow_html=True)
                for name, score in lg['evidenz'][:3]:
                    if score > 0:
                        st.markdown(f"""
                            <div class="xai-feature-row"><span>{name}</span><span>{score:.1f}%</span></div>
                            <div class="xai-bar-bg"><div class="xai-bar-fill" style="width: {score}%;"></div></div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("Warte auf Sensorsignale des digitalen Zwillings...")

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

# --- 7. APPLIKATIONS-STEUERUNG & HARDWARE-SCHLEIFE ---
st.divider()
b1, b2 = st.columns(2)
if b1.button("▶ START / PAUSE", use_container_width=True): 
    s['active'] = not s['active']
    st.rerun()
if b2.button("🔄 RESET WERKZEUG", use_container_width=True):
    st.session_state.twin = {
        'zyklus': 0.0, 'zyklen_anzahl': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0, 'drehzahl': 0.0,
        'seed': np.random.RandomState(42)
    }
    st.rerun()

if s['active']:
    time.sleep(max(0.01, taktzeit / 1000.0 if taktzeit > 0 else 0.05))
    st.rerun()
