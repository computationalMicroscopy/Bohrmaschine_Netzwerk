import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-END INDUSTRIAL STYLING ---
st.set_page_config(layout="wide", page_title="KI-Zerspanungs-Plattform TwinPro V5.2", page_icon="⚙️")

# --- INITIALISIERUNG & STATE-MACHINE ---
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

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Live-Prozessparameter")
    mat_name = st.selectbox("Ausgewählter Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    
    vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 30, 350, 100)
    f = st.slider("Vorschub pro Umdrehung f [mm/U]", 0.05, 0.60, 0.15)
    d = st.slider("Bohrer-Durchmesser d [mm]", 5.0, 32.0, 12.0)
    kuehlung = st.toggle("Kühlschmierstoff (KSS) aktiv", value=True)
    
    st.divider()
    st.header("🎛️ Zeitskalierung & Sensoren")
    schrittweite = st.number_input("Zeitskalierungsfaktor (Beliebig hoch)", min_value=1, max_value=1000000, value=5)
    taktzeit = st.select_slider("Aktualisierungsintervall (ms)", options=[500, 200, 100, 0], value=100)
    
    sensor_temp_gain = st.slider("Temperatursensor-Empfindlichkeit", 0.5, 2.5, 1.0, step=0.1)
    sensor_vibr_gain = st.slider("Vibrationssensor-Verstärkung (Gain)", 0.5, 3.0, 1.0, step=0.1)
    noise_level = st.slider("Rausch-Amplitude (Vibration)", 0.1, 2.0, 0.5)

# DYNAMISCHE ANIMATIONS-ZEITBERECHNUNG
basis_zyklus_zeit = 2.5 
dynamische_animations_dauer = max(0.05, basis_zyklus_zeit / schrittweite)

# --- GLOBAL DYNAMIC CSS STYLES ---
st.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;800&display=swap');
    
    .stApp {{ 
        background-color: #06090e; 
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }}
    
    label, .stSlider, .stSelectbox, .stToggle {{ 
        font-size: 1.25rem !important; 
        font-weight: 600 !important; 
        color: #f0f6fc !important;
    }}
    .stMarkdown p {{ font-size: 1.15rem !important; line-height: 1.6; }}
    
    .main-title {{
        font-size: 2.8rem; font-weight: 800; color: #f0f6fc;
        text-align: center; margin-bottom: 30px; padding-bottom: 20px;
        border-bottom: 1px solid rgba(240, 246, 252, 0.1);
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .glass-card {{
        background: rgba(16, 22, 30, 0.85); 
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 12px; padding: 20px; margin-bottom: 15px; text-align: center;
        box-shadow: 0 4px 25px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(88, 166, 255, 0.5);
        box-shadow: 0 4px 30px rgba(88, 166, 255, 0.15);
    }}
    .val-title {{ font-size: 1.05rem; color: #8b949e; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }}
    .val-main {{ font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 800; color: #f0f6fc; margin-top: 5px; display: block; }}
    
    .xai-container {{ height: 400px; overflow-y: auto; padding-right: 5px; }}
    .xai-card {{
        background: #0d1117; border-left: 6px solid #58a6ff;
        padding: 18px; border-radius: 8px; margin-bottom: 12px;
        border-top: 1px solid rgba(255,255,255,0.02);
    }}
    .xai-feature-row {{ display: flex; justify-content: space-between; font-size: 1.1rem; color: #c9d1d9; margin-top: 6px; font-weight: 500;}}
    .xai-bar-bg {{ background: #21262d; height: 8px; width: 100%; border-radius: 4px; margin-bottom: 8px; overflow:hidden;}}
    .xai-bar-fill {{ background: linear-gradient(90deg, #58a6ff, #1f6feb); height: 100%; border-radius: 4px; }}
    .reason-text {{ color: #f0f6fc; font-size: 1.25rem; margin-top: 6px; font-weight: 700; }}
    .sensor-snapshot {{ font-size: 1.0rem; color: #8b949e; margin-top: 6px; font-family: 'JetBrains Mono', monospace; border-bottom: 1px solid #30363d; padding-bottom: 6px;}}
    .action-text {{ color: #ff7b72; font-weight: bold; font-size: 1.15rem; margin-top: 8px; border-top: 1px solid rgba(48, 54, 61, 0.5); padding-top: 8px; }}
    .diag-badge {{ background: #23426f; color: #58a6ff; border: 1px solid #388bfd; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;}}
    
    .emergency-alert {{
        background: linear-gradient(90deg, #da3633, #8a1f1d); color: white; padding: 20px; border-radius: 10px; 
        font-weight: 800; text-align: center; margin-bottom: 25px; font-size: 1.5rem;
        box-shadow: 0 0 30px rgba(218, 54, 51, 0.4); border: 1px solid #f85149;
    }}

    @keyframes helical_spin {{
        0% {{ background-position: 0px 0px, 0px 0px; }}
        100% {{ background-position: 0px -180px, 0px 0px; }}
    }}
    
    @keyframes tool_feed_optimized {{
        0% {{ transform: translateY(-35px); }}      
        15% {{ transform: translateY(-35px); }}     
        25% {{ transform: translateY(-2px); }}      
        65% {{ transform: translateY(22px); }}      
        75% {{ transform: translateY(22px); }}      
        90% {{ transform: translateY(-35px); }}     
        100% {{ transform: translateY(-35px); }}
    }}

    @keyframes vfx_sync_gate {{
        0%, 24% {{ opacity: 0; transform: scale(0.5); }}
        25% {{ opacity: 1; transform: scale(1); }}
        65% {{ opacity: 1; transform: scale(1); }}
        66%, 75% {{ opacity: 0.2; transform: scale(0.8); }} 
        76%, 100% {{ opacity: 0; transform: scale(0); }}   
    }}

    @keyframes led_pulse {{
        0%, 100% {{ box-shadow: 0 0 10px var(--led-color), inset 0 0 5px var(--led-color); }}
        50% {{ box-shadow: 0 0 26px var(--led-color), inset 0 0 12px var(--led-color); }}
    }}
    @keyframes strobe_crit {{
        0%, 100% {{ background: #ff0000; box-shadow: 0 0 25px #ff0000; }}
        50% {{ background: #200000; box-shadow: 0 0 2px #200000; }}
    }}
    </style>
""")

st.html('<div class="main-title">🚀 Next-Gen KI-Zerspanungslabor & XAI-Plattform</div>')

# --- 2. DETERMINISTISCHE REAL-SENSOR KI DIAGNOSE ENGINE ---
def compute_sensor_diagnostics(current_vals, settings, integrity, kuehlung, m):
    M, T, V, F = current_vals['M'], current_vals['T'], current_vals['V'], current_vals['F']
    vc, f, d = settings['vc'], settings['f'], settings['d']
    
    crit_torque = 0.12 * (d ** 3) 
    crit_force = 320 * (d ** 2)
    t_crit = m['t_crit']
    
    evidenz = {
        "Mechanische Torsions-Überlast": 0.0,
        "Thermische Gefüge-Erweichung": 0.0,
        "Regeneratives Rattern (Resonanz)": 0.0,
        "Aufbauschneidenbildung (Adhäsion)": 0.0,
        "Axiale Schaft-Knickung (Vorschub zu hoch)": 0.0,
        "Kühlungs-Abriss (Thermoschock-Gefahr)": 0.0,
        "Spanstau / Spanflächen-Verstopfung": 0.0,
        "Mikro-Bröckelung der Schneidkante": 0.0,
        "Schnittdaten-Unterforderung (Kaltverfestigung)": 0.0,
        "Abrasiver Freiflächenverschleiß (Normal)": 0.0,
        "Spindellager-Überlastung (Vibration resonant)": 0.0,
        "Extrem-Vorschub-Stauchung": 0.0,
        "Kratzender Schnitt (Zentrierfehler)": 0.0,
        "KSS-Verdampfung (Kavitation)": 0.0,
        "Erhöhte Reibung durch Verschleißmarkenbreite": 0.0
    }
    
    if M > crit_torque * 0.85: evidenz["Mechanische Torsions-Überlast"] = min(100.0, (M / crit_torque) * 100.0)
    if T > t_crit * 0.85: evidenz["Thermische Gefüge-Erweichung"] = min(100.0, (T / t_crit) * 100.0)
    if V > 5.0: evidenz["Regeneratives Rattern (Resonanz)"] = min(100.0, (V / 8.0) * 100.0)
    if vc < 60 and f > 0.12 and T < 200: evidenz["Aufbauschneidenbildung (Adhäsion)"] = 85.0 + (f * 20.0)
    if F > crit_force * 0.85: evidenz["Axiale Schaft-Knickung (Vorschub zu hoch)"] = min(100.0, (F / crit_force) * 100.0)
    if not kuehlung and T > 180: evidenz["Kühlungs-Abriss (Thermoschock-Gefahr)"] = min(100.0, (T / 250.0) * 60.0 + 40.0)
    if M > crit_torque * 0.7 and V > 4.0: evidenz["Spanstau / Spanflächen-Verstopfung"] = 75.0
    if V > 6.0 and integrity < 60: evidenz["Mikro-Bröckelung der Schneidkante"] = 80.0
    if vc > 150 and f < 0.08: evidenz["Schnittdaten-Unterforderung (Kaltverfestigung)"] = 70.0
    if integrity < 85: evidenz["Abrasiver Freiflächenverschleiß (Normal)"] = (100.0 - integrity)
    if V > 7.0 and M < crit_torque * 0.4: evidenz["Spindellager-Überlastung (Vibration resonant)"] = 90.0
    if f > 0.45: evidenz["Extrem-Vorschub-Stauchung"] = 85.0
    if F > crit_force * 0.6 and M < crit_torque * 0.3: evidenz["Kratzender Schnitt (Zentrierfehler)"] = 65.0
    if kuehlung and T > 100: evidenz["KSS-Verdampfung (Kavitation)"] = min(95.0, (T / 100.0) * 45.0)
    if integrity < 40 and M > crit_torque * 0.6: evidenz["Erhöhte Reibung durch Verschleißmarkenbreite"] = 75.0

    sorted_evidenz = sorted(evidenz.items(), key=lambda x: x[1], reverse=True)
    top_reason = sorted_evidenz[0][0]
    
    mapping = {
        "Mechanische Torsions-Überlast": {"diag": "CRITICAL TORQUE", "exp": f"Das gemessene Drehmoment von {M:.1f} Nm übersteigt die Scherspannungsgrenze des Bohrers.", "act": f"MINDERN SIE DEN VORSCHUB: Reduzieren Sie f sofort um 30% auf {f*0.7:.2f} mm/U."},
        "Thermische Gefüge-Erweichung": {"diag": "THERMAL OVERLOAD", "exp": f"Die Schnittkantentemperatur ({T:.0f}°C) hat die Anlasstemperatur überschritten. Akute Gefügeerweichung!", "act": f"REDUZIEREN SIE REIBUNG: Schnittgeschwindigkeit vc um 25% auf {int(vc*0.75)} m/min drosseln."},
        "Regeneratives Rattern (Resonanz)": {"diag": "RESONANT CHATTER", "exp": f"Selbsterregte Schwingungen ({V:.1f} mm/s) stören den Spanbildungsprozess massiv.", "act": f"DREHZAHL-SHIFT ERFORDERLICH: Ändern Sie vc um +15% auf {int(vc*1.15)} m/min, um Resonanz zu brechen."},
        "Aufbauschneidenbildung (Adhäsion)": {"diag": "ADHESION DETECTED", "exp": "Die Schnittgeschwindigkeit ist zu gering. Werkstoffteilchen verschweißen kalt auf der Schneide.", "act": f"ERHÖHEN SIE vc: Fahren Sie vc um 20% hoch ({int(vc*1.2)} m/min) für eine thermodynamische Stabilisierung."},
        "Axiale Schaft-Knickung (Vorschub zu hoch)": {"diag": "AXIAL BUCKLED", "exp": f"Die axiale Vorschubkraft ({F:.0f} N) nähert sich der kritischen Knicklast nach Euler.", "act": f"VORSCHUB HALBIEREN: Reduzieren Sie f umgehend auf {f*0.5:.2f} mm/U gegen Kernbruch."},
        "Kühlungs-Abriss (Thermoschock-Gefahr)": {"diag": "TRIBOLOGY FAILURE", "exp": "Schlagartiger Temperaturanstieg mangels KSS-Medium. Direktes Zuschalten erzeugt Thermoschock-Risse.", "act": "ZYKLUS UNTERBRECHEN: Vorschub stoppen, Werkzeug rotierend aus der Bohrung fahren und an Luft kühlen."},
        "Spanstau / Spanflächen-Verstopfung": {"diag": "CHIP CLOGGING", "exp": "Drehmoment und Vibrationen steigen simultan. Späne blockieren die Spannuten.", "act": "PICK-FEEDING AKTIVIEREN: Fahren Sie einen Entspanungszyklus mit verkürzter Bohrtiefe pro Hub."},
        "Mikro-Bröckelung der Schneidkante": {"diag": "MICRO-CHIPPING", "exp": "Hochfrequente Vibrationsspitzen erzeugten feine Ausbrüche an der Schneidecke.", "act": "WERKZEUGPRÜFUNG: Vorschub f leicht reduzieren und Schneidkanten beim nächsten Stopp optisch prüfen."},
        "Schnittdaten-Unterforderung (Kaltverfestigung)": {"diag": "LOW ENGAGEMENT", "exp": "Der Vorschub ist zu gering, die Schneide schabt mehr als sie schneidet. Werkstoff verfestigt sich.", "act": f"SPANSTÄRKE ERHÖHEN: Steigern Sie den Vorschub f auf mindestens {max(0.12, f*1.5):.2f} mm/U."},
        "Abrasiver Freiflächenverschleiß (Normal)": {"diag": "ABRASIVE WEAR", "exp": f"Fortschreitender Verschleiß durch Carbide im Material. Restintegrität liegt bei {integrity:.1f}%.", "act": "STANDZEIT-MONITORING: Der Prozess läuft stabil. Werkzeugwechsel beim nächsten Wartungsfenster planen."},
        "Spindellager-Überlastung (Vibration resonant)": {"diag": "SPINDLE VIBRATION", "exp": "Starke Vibrationen trotz niedriger Drehmomentlast weisen auf eine Unwucht der Werkzeugspindel hin.", "act": "WUCHT PRÜFEN: Drehzahl absenken. Werkzeugaufnahme auf Achsversatz oder Verschmutzung untersuchen."},
        "Extrem-Vorschub-Stauchung": {"diag": "HIGH FEED HEAVE", "exp": "Der gewählte Vorschub zwingt die Querschneide zu extremer plastischer Deformation des Materials.", "act": "VORSCHUB ANPASSEN: Reduzieren Sie f auf maximal 0.30 mm/U, um axiale Kernkompression zu mindern."},
        "Kratzender Schnitt (Zentrierfehler)": {"diag": "CENTERING ERROR", "exp": "Hohe Axialkraft bei minimalem Drehmoment deutet darauf hin, dass die Zentrierspitze reibt.", "act": "ANBOHRUNG PRÜFEN: Vorbohrung kontrollieren. Spitzenwinkel des Anbohrers muss größer sein."},
        "KSS-Verdampfung (Kavitation)": {"diag": "COOLANT BOILING", "exp": "Temperatur liegt über 100°C trotz aktivem KSS. Das Kühlmedium verdampft direkt an der Wirkstelle.", "act": "DRUCK ERHÖHEN: Erhöhen Sie den KSS-Druck über interne Kühlung (IKZ), um das Medium flüssig zu halten."},
        "Erhöhte Reibung durch Verschleißmarkenbreite": {"diag": "FRICTION ELEVATION", "exp": "Das Drehmoment kriecht nach oben, bedingt durch vergrößerte Reibflächen verbrauchter Schneiden.", "act": "WERKZEUG WECHSELN: Die Standzeitgrenze ist nahezu erreicht. Bereiten Sie den Tausch des Bohrers vor."}
    }
    
    res = mapping.get(top_reason, {"diag": "NOMINAL", "exp": "System läuft im stabilen Bereich.", "act": "Keine Korrekturdaten nötig."})
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
    norm_vibr = s['vibration'] / 8.0
    norm_wear = (100.0 - s['integritaet']) / 100.0
    
    exp_report, evidenz_list = compute_sensor_diagnostics(
        {'M': s['drehmoment'], 'T': s['thermik'], 'V': s['vibration'], 'F': s['vorschubkraft']},
        {'vc': vc, 'f': f, 'd': d}, s['integritaet'], kuehlung, m
    )
    
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
    
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'info': exp_report, 'evidenz': evidenz_list})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration'], 'p': s['leistung'], 'm': s['drehmoment'], 'f': s['vorschubkraft']})

# --- CRASH-DETEKTION ---
if s['broken']: st.html('<div class="emergency-alert">💥 STRUKTURELLER WERKZEUGBRUCH! Schaft zerstört.</div>')
if s['stall']: st.html('<div class="emergency-alert">⚠️ MOTOR-STALL: Spindel blockiert (7.5 kW).</div>')

# --- 4. DYNAMISCHE GEOMETRIE-SKALIERUNG DER VFX ENGINE ---
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

col_animation, col_metrics = st.columns([1.5, 4])

with col_animation:
    t_val = s['thermik']
    integ_val = s['integritaet']
    
    if s['broken'] or s['stall']:
        led_style = "animation: strobe_crit 0.2s infinite;"
    elif not s['active']:
        led_style = "background: #555; box-shadow: 0 0 5px #333; --led-color: #555;"
    else:
        if s['risk'] > 0.7: color, hex_val = "rgba(255, 68, 0, 1)", "#ff4400"
        elif kuehlung: color, hex_val = "rgba(0, 180, 255, 1)", "#00b4ff"
        else: color, hex_val = "rgba(46, 164, 79, 1)", "#2ea44f"
        led_style = f"background: {hex_val}; --led-color: {color}; animation: led_pulse 1s infinite ease-in-out;"

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

    wear_factor = (100.0 - integ_val) / 100.0
    tip_color = f"linear-gradient(to bottom, #444 0%, {tip_base} 65%, rgba(15,15,15,{wear_factor:.2f}) 100%)" if not s['broken'] else "#1a1a1a"
    surface_glow = f"rgba(255, 60, 0, {min(0.9, (t_val-100)/500)})" if t_val > 100 else "rgba(0,0,0,0)"

    synchronized_vfx = ""
    if s['active']:
        synchronized_vfx += f"""
        <div style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; animation: vfx_sync_gate {dynamische_animations_dauer}s infinite ease-in-out; pointer-events: none;">
            <div style="position: absolute; left: {int(8*scale_factor)}px; bottom: 35px; width: {max(3, int(6*scale_factor))}px; height: 4px; background: {m['color']}; border-radius:2px; animation: chip_spray_left 0.1s infinite linear;"></div>
            <div style="position: absolute; right: {
