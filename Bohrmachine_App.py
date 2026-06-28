
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & HIGH-END INDUSTRIAL STYLING ---
st.set_page_config(layout="wide", page_title="Bohrersimulation", page_icon="⚙️")

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

# DYNAMISCHE ANIMATIONS-ZEITBERECHNUNG (Verhindert den Stroboskop-Effekt)
basis_zyklus_zeit = 2.5 
dynamische_animations_dauer = max(0.05, basis_zyklus_zeit / schrittweite)

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
    
    .xai-container {{ height: 600px; overflow-y: auto; padding-right: 5px; }}
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

    @keyframes chip_spray_left {{
        0% {{ transform: translate(0, 0) scale(1.3) rotate(0deg); opacity: 1; }}
        100% {{ transform: translate(-90px, -45px) scale(0.1) rotate(-720deg); opacity: 0; }}
    }}
    @keyframes chip_spray_right {{
        0% {{ transform: translate(0, 0) scale(1.3) rotate(0deg); opacity: 1; }}
        100% {{ transform: translate(90px, -45px) scale(0.1) rotate(720deg); opacity: 0; }}
    }}
    @keyframes smoke_rise {{
        0% {{ transform: translate(-50%, 0px) scale(0.4); opacity: 0; }}
        20% {{ opacity: 0.5; filter: blur(3px); }}
        100% {{ transform: translate(-50%, -80px) scale(3.8); opacity: 0; filter: blur(9px); }}
    }}
    @keyframes kss_flood_left {{
        0% {{ transform: translate(-50px, -70px) scaleY(1) rotate(42deg); opacity: 0.6; }}
        100% {{ transform: translate(-3px, -2px) scaleX(0.3) rotate(42deg); opacity: 1; }}
    }}
    @keyframes kss_flood_right {{
        0% {{ transform: translate(50px, -70px) scaleY(1) rotate(-42deg); opacity: 0.6; }}
        100% {{ transform: translate(3px, -2px) scaleX(0.3) rotate(-42deg); opacity: 1; }}
    }}
    @keyframes kss_mist {{
        0% {{ transform: translate(-50%, -5px) scale(0.7); opacity: 0.4; }}
        100% {{ transform: translate(-50%, -35px) scale(2.0); opacity: 0; }}
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

st.html('<div class="main-title">Bohrersimulation & XAI-Plattform</div>')

# --- 2. DETERMINISTISCHE REAL-SENSOR KI DIAGNOSE ENGINE ---
def compute_sensor_diagnostics(current_vals, settings, integrity, kuehlung, m):
    M, T, V, F = current_vals['M'], current_vals['T'], current_vals['V'], current_vals['F']
    vc, f, d = settings['vc'], settings['f'], settings['d']
    
    crit_torque = 0.12 * (d ** 3) 
    crit_force = 320 * (d ** 2)
    t_crit = m['t_crit']
    
    # 15+ Hochpräzise deterministische Zerspanungszustände
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
    
    # Physikalische Scoring-Logiken basierend auf den exakten Sensor-Interaktionen
    if M > crit_torque * 0.85:
        evidenz["Mechanische Torsions-Überlast"] = min(100.0, (M / crit_torque) * 100.0)
    if T > t_crit * 0.85:
        evidenz["Thermische Gefüge-Erweichung"] = min(100.0, (T / t_crit) * 100.0)
    if V > 5.0:
        evidenz["Regeneratives Rattern (Resonanz)"] = min(100.0, (V / 8.0) * 100.0)
    if vc < 60 and f > 0.12 and T < 200:
        evidenz["Aufbauschneidenbildung (Adhäsion)"] = 85.0 + (f * 20.0)
    if F > crit_force * 0.85:
        evidenz["Axiale Schaft-Knickung (Vorschub zu hoch)"] = min(100.0, (F / crit_force) * 100.0)
    if not kuehlung and T > 180:
        evidenz["Kühlungs-Abriss (Thermoschock-Gefahr)"] = min(100.0, (T / 250.0) * 60.0 + 40.0)
    if M > crit_torque * 0.7 and V > 4.0:
        evidenz["Spanstau / Spanflächen-Verstopfung"] = 75.0
    if V > 6.0 and integrity < 60:
        evidenz["Mikro-Bröckelung der Schneidkante"] = 80.0
    if vc > 150 and f < 0.08:
        evidenz["Schnittdaten-Unterforderung (Kaltverfestigung)"] = 70.0
    if integrity < 85:
        evidenz["Abrasiver Freiflächenverschleiß (Normal)"] = (100.0 - integrity)
    if V > 7.0 and M < crit_torque * 0.4:
        evidenz["Spindellager-Überlastung (Vibration resonant)"] = 90.0
    if f > 0.45:
        evidenz["Extrem-Vorschub-Stauchung"] = 85.0
    if F > crit_force * 0.6 and M < crit_torque * 0.3:
        evidenz["Kratzender Schnitt (Zentrierfehler)"] = 65.0
    if kuehlung and T > 100:
        evidenz["KSS-Verdampfung (Kavitation)"] = min(95.0, (T / 100.0) * 45.0)
    if integrity < 40 and M > crit_torque * 0.6:
        evidenz["Erhöhte Reibung durch Verschleißmarkenbreite"] = 75.0

    # Sortieren nach Stärke der Indikation
    sorted_evidenz = sorted(evidenz.items(), key=lambda x: x[1], reverse=True)
    top_reason = sorted_evidenz[0][0]
    
    # Zuweisung hochrealistischer Industrie-Handlungsempfehlungen
    mapping = {
        "Mechanische Torsions-Überlast": {
            "diag": "CRITICAL TORQUE",
            "exp": f"Das gemessene Drehmoment von {M:.1f} Nm übersteigt die Scherspannungsgrenze des Bohrers.",
            "act": f"MINDERN SIE DEN VORSCHUB: Reduzieren Sie f sofort um 30% auf {f*0.7:.2f} mm/U. Ein Senken von vc entlastet die Torsion nicht!"
        },
        "Thermische Gefüge-Erweichung": {
            "diag": "THERMAL OVERLOAD",
            "exp": f"Die Schnittkantentemperatur ({T:.0f}°C) hat die Anlasstemperatur des Schneidstoffs überschritten. Akute Härteerweichung!",
            "act": f"REDUZIEREN SIE DIE REIBUNGSLEISTUNG: Schnittgeschwindigkeit vc um 25% auf {int(vc*0.75)} m/min drosseln und KSS-Volumenstrom erhöhen."
        },
        "Regeneratives Rattern (Resonanz)": {
            "diag": "RESONANT CHATTER",
            "exp": f"Selbsterregte Schwingungen ({V:.1f} mm/s) stören den Spanbildungsprozess. Schneide droht unkontrolliert zu splittern.",
            "act": f"DREHZAHL-SHIFT ERFORDERLICH: Brechen Sie die harmonische Resonanzwelle durch Änderung von vc um +15% auf {int(vc*1.15)} m/min."
        },
        "Aufbauschneidenbildung (Adhäsion)": {
            "diag": "ADHESION DETECTED",
            "exp": "Die Schnittgeschwindigkeit ist zu gering für die herrschende Pressung. Werkstoffteilchen verschweißen kalt auf der Schneide.",
            "act": f"ERHÖHEN SIE vc: Fahren Sie vc um 20% hoch ({int(vc*1.2)} m/min), um die Fließzone thermodynamisch zu stabilisieren."
        },
        "Axiale Schaft-Knickung (Vorschub zu hoch)": {
            "diag": "AXIAL BUCKLED",
            "exp": f"Die axiale Vorschubkraft ({F:.0f} N) nähert sich der kritischen Knicklast nach Euler.",
            "act": f"VORSCHUBKRAFT HALBIEREN: Reduzieren Sie f umgehend auf {f*0.5:.2f} mm/U, um ein Auswandern oder Brechen des Kerns zu verhindern."
        },
        "Kühlungs-Abriss (Thermoschock-Gefahr)": {
            "diag": "TRIBOLOGY FAILURE",
            "exp": "Schlagartiger Temperaturanstieg mangels KSS-Medium. Direktes Zuschalten bei dieser Hitze erzeugt Risse durch Thermoschock.",
            "act": "ZYKLUS UNTERBRECHEN: Vorschub stoppen, Werkzeug rotierend aus der Bohrung fahren und langsam an der Luft abkühlen lassen."
        },
        "Spanstau / Spanflächen-Verstopfung": {
            "diag": "CHIP CLOGGING",
            "exp": "Drehmoment und Vibrationen steigen simultan. Späne werden in den Nuten nicht sauber abgeführt und blockieren den Kanal.",
            "act": f"ENTSCHACHTELN & ENTSPANEN: Fahren Sie einen Entspanungszyklus (Pick-Feeding) mit verkürzter Bohrtiefe pro Hub."
        },
        "Mikro-Bröckelung der Schneidkante": {
            "diag": "MICRO-CHIPPING",
            "exp": "Hochfrequente Vibrationsspitzen haben zu feinen Ausbrüchen an der Schneidecke geführt. Erhöhter Verschleißprozess.",
            "act": "WERKZEUGPRÜFUNG: Vorschub f leicht reduzieren. Bei der nächsten Gelegenheit Schneidkanten unter dem Mikroskop optisch prüfen."
        },
        "Schnittdaten-Unterforderung (Kaltverfestigung)": {
            "diag": "LOW ENGAGEMENT",
            "exp": "Der Vorschub ist zu gering, die Schneide schabt mehr als sie schneidet. Der Werkstoff verfestigt sich plastisch.",
            "act": f"SPANSTÄRKE ERHÖHEN: Steigern Sie den Vorschub f auf mindestens {max(0.12, f*1.5):.2f} mm/U, um unter die Verfestigungszone zu kommen."
        },
        "Abrasiver Freiflächenverschleiß (Normal)": {
            "diag": "ABRASIVE WEAR",
            "exp": f"Fortschreitender Verschleiß durch harte Carbide im Material Gefüge. Restintegrität liegt bei {integrity:.1f}%.",
            "act": "STANDZEIT-MONITORING: Der Prozess läuft stabil. Werkzeugwechsel beim nächsten planmäßigen Wartungsfenster einplanen."
        },
        "Spindellager-Überlastung (Vibration resonant)": {
            "diag": "SPINDLE VIBRATION",
            "exp": "Starke Vibrationen trotz niedriger Drehmomentlast weisen auf eine Unwucht oder einen Lagerschaden der Werkzeugspindel hin.",
            "act": "WUCHT PRÜFEN: Drehzahl absenken. Aufnahme auf Verschmutzung oder Achsversatz im Spindelkopf untersuchen."
        },
        "Extrem-Vorschub-Stauchung": {
            "diag": "HIGH FEED HEAVE",
            "exp": "Der gewählte Vorschub zwingt die Querschneide zu extremer plastischer Deformation des Materials. Enorme axiale Pressung.",
            "act": f"VORSCHUB ANPASSEN: Reduzieren Sie f auf maximal 0.30 mm/U, um die mechanische Kernkompression zu mindern."
        },
        "Kratzender Schnitt (Zentrierfehler)": {
            "diag": "CENTERING ERROR",
            "exp": "Hohe Axialkraft bei minimalem Drehmoment deutet darauf hin, dass die Zentrierspitze reibt, ohne dass die Hauptschneiden greifen.",
            "act": "ANBOHRUNG PRÜFEN: Vorbohrung/Anzentrierung kontrollieren. Spitzenwinkel des Anbohrers muss größer sein als der des Hauptbohrers."
        },
        "KSS-Verdampfung (Kavitation)": {
            "diag": "COOLANT BOILING",
            "exp": "Temperatur liegt über 100°C trotz aktivem KSS. Das Kühlmedium verdampft direkt an der Wirkstelle und verliert die Kühlwirkung.",
            "act": "DRUCK ERHÖHEN: Erhöhen Sie den KSS-Druck (IKZ) von Außenkühlung auf interne Kühlung, um das Medium unter Druck flüssig zu halten."
        },
        "Erhöhte Reibung durch Verschleißmarkenbreite": {
            "diag": "FRICTION ELEVATION",
            "exp": "Das Drehmoment kriecht nach oben, bedingt durch die vergrößerte Reibfläche an den verbrauchten Schneidkanten.",
            "act": "WERKZEUG WECHSELN: Die Standzeitgrenze ist nahezu erreicht. Bereiten Sie den Tausch des Bohrers vor."
        }
    }
    
    res = mapping.get(top_reason)
    res["snapshot"] = f"M: {M:.1f}Nm | T: {T:.0f}°C | V: {V:.1f}mm/s | F_f: {F:.0f}N"
    return res, sorted_evidenz

# --- PHYSIK-ENGINE ---
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
    
    # Aufruf der neuen Sensor-Diagnose-Engine
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
if s['broken']:
    st.html('<div class="emergency-alert">💥 STRUKTURELLER WERKZEUGBRUCH! Schaft zerstört.</div>')
if s['stall']:
    st.html('<div class="emergency-alert">⚠️ MOTOR-STALL: Spindel blockiert (7.5 kW).</div>')

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
            <div style="position: absolute; left: 8px; bottom: 35px; width: 6px; height: 4px; background: {m['color']}; border-radius:2px; animation: chip_spray_left 0.1s infinite linear;"></div>
            <div style="position: absolute; right: 8px; bottom: 35px; width: 5px; height: 3px; background: {m['color']}; border-radius:1px; animation: chip_spray_right 0.09s infinite linear; animation-delay: 0.03s;"></div>
        """
        if kuehlung:
            synchronized_vfx += """
            <div style="position: absolute; width: 3px; height: 75px; background: rgba(180, 240, 255, 0.7); filter: blur(0.5px); animation: kss_flood_left 0.12s infinite linear; transform-origin: top left;"></div>
            <div style="position: absolute; width: 3px; height: 75px; background: rgba(180, 240, 255, 0.7); filter: blur(0.5px); animation: kss_flood_right 0.12s infinite linear; transform-origin: top right;"></div>
            <div style="position: absolute; left: 50%; bottom: 32px; width: 30px; height: 15px; background: rgba(200, 235, 255, 0.15); filter: blur(4px); border-radius: 50%; animation: kss_mist 0.25s infinite ease-out;"></div>
            """
        if t_val > 220:
            synchronized_vfx += '<div style="position: absolute; left: 50%; bottom: 35px; width: 18px; height: 18px; background: rgba(240,240,240,0.15); filter: blur(7px); border-radius: 50%; animation: smoke_rise 0.3s infinite linear;"></div>'
        if t_val > 420:
            synchronized_vfx += """
            <div style="position: absolute; left: 12px; bottom: 35px; width: 3px; height: 3px; background: #ffcc00; box-shadow:0 0 5px #ff3300; border-radius:50%; animation: chip_spray_left 0.05s infinite linear;"></div>
            <div style="position: absolute; right: 12px; bottom: 35px; width: 3px; height: 3px; background: #fffa00; box-shadow:0 0 5px #ff3300; border-radius:50%; animation: chip_spray_right 0.05s infinite linear;"></div>
            """
        synchronized_vfx += "</div>"

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
        anim_feed = f"tool_feed_optimized {dynamische_animations_dauer}s infinite ease-in-out" if s['active'] else "none"
        
        drill_render = f"""
        <div style="animation: {anim_spin}; width: 44px; height: 115px; 
                    background: linear-gradient(90deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 20%, rgba(0,0,0,0.4) 70%, rgba(0,0,0,0.7) 100%),
                                linear-gradient(to right, transparent 40%, rgba(255,255,255,0.15) 50%, transparent 60%),
                                repeating-linear-gradient(135deg, #12161b 0px, #12161b 12px, #3a3a3a 16px, #777 22px, #3a3a3a 26px, #12161b 38px); 
                    background-size: 100% 100%, 100% 100%, 100% 45px; box-shadow: inset 4px 0 10px rgba(0,0,0,0.75);"></div>
        <div style="background: {tip_color}; box-shadow: {glow_effect}; width: 44px; height: 21px; clip-path: polygon(0% 0%, 100% 0%, 50% 100%); margin-top: -1px;"></div>
        """
        status_label = "<span style='color:#2ea44f; font-weight:900;'>ROTATION LIVE</span>" if s['active'] else "<span style='color:#8b949e;'>STANDBY</span>"
        if s['stall']: status_label = "<span style='color:#e3b341; font-weight:900;'>STALL / BLOCKIERT</span>"

    st.html(f"""
        <div class="glass-card" style="padding: 20px; height: 100%; min-height: 340px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; background: #07090e; position: relative; overflow: hidden; border: 1px solid #21262d;">
            <span class="val-title" style="color: #bc8cff; font-size:1.05rem;">Bohrer - Animation</span>
            
            <div style="position: relative; display: flex; flex-direction: column; align-items: center; z-index:4;">
                <div style="width: 76px; height: 30px; background: linear-gradient(90deg, #161b22 0%, #485260 50%, #161b22 100%); border-radius: 4px 4px 0 0; border: 1px solid #30363d;"></div>
                <div style="{led_style} width: 66px; height: 6px; margin-top: -1px; border-radius: 0 0 2px 2px; transition: all 0.3s;"></div>
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

# --- UX-DASHBOARD-KPI PANELS ---
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

# --- TABS ---
t1, t2 = st.tabs(["📈 Echtzeit-Telemetrie & Oszilloskop", "🔬 Prädiktives Was-Wäre-Wenn Simulationslabor"])

with t1:
    col_graph, col_log = st.columns([2.1, 1])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Strukturelle Werkzeugintegrität (%)", "Kinetische Lastprofile", "Prozessdynamik"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#2ea44f', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['m'], line=dict(color='#bc8cff', width=2.5)), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['f']/100.0, line=dict(color='#1f6feb', width=2, dash='dash')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#a371f7', width=2.5)), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#ff7b72', width=2.5)), 3, 1)
            fig.update_layout(height=580, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
    with col_log:
        st.markdown("### Sensorbasierte KI-Ursachendiagnose (XAI)")
        if s['logs']:
            html_str = '<div class="xai-container">'
            for l in s['logs'][:10]:
                bars = "".join([f'<div class="xai-feature-row"><span>{e[0]}</span><span>{e[1]:.1f}%</span></div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{e[1]}%"></div></div>' for e in l['evidenz'][:3] if e[1] > 0])
                html_str += f"""
                <div class="xai-card">
                    <div style="display:flex; align-items:center; margin-bottom:6px;">
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

with t2:
    st.markdown("### 🧪 Labor-Simulationsraum")
    col_inputs, col_outputs = st.columns([1, 1])
    with col_inputs:
        lab_mat = st.selectbox("Zu bearbeitender Labor-Werkstoff", list(MATERIALIEN.keys()), key="lab_mat")
        lm = MATERIALIEN[lab_mat]
        lab_vc = st.slider("Schnittgeschwindigkeit vc [m/min]", 30, 350, 100, key="lab_vc")
        lab_f = st.slider("Gewählter Vorschub f [mm/U]", 0.05, 0.60, 0.15, key="lab_f")
        lab_d = st.slider("Bohrer-Nenndurchmesser d [mm]", 5.0, 32.0, 12.0, key="lab_d")
        lab_kss = st.toggle("Kühlschmierstoff-Zufuhr aktiv (KSS)", value=True, key="lab_kss")
        lab_integ = st.slider("Aktuelle Werkzeug-Restlebensdauer [%]", 0.0, 100.0, 100.0, key="lab_integ")
        lab_vibr_override = st.slider("Künstlich überlagerte Schwingung [mm/s]", 0.1, 15.0, 0.4, key="lab_vibr")

    with col_outputs:
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
        
        # Aufruf der Diagnose-Engine innerhalb des Labors
        lab_report, lab_evidenz_list = compute_sensor_diagnostics(
            {'M': l_torque, 'T': l_temp, 'V': lab_vibr_override * sensor_vibr_gain, 'F': l_force},
            {'vc': lab_vc, 'f': lab_f, 'd': lab_d}, lab_integ, lab_kss, lm
        )
        
        l_combined_score = max([l_torque/c_t, l_force/c_f, l_temp/lm['t_crit']]) + ((100.0 - lab_integ) / 100.0)**2 * 0.95
        lab_risk = np.clip(0.15 + (l_combined_score - 0.75) * 2.3 if l_combined_score >= 0.75 else l_combined_score * 0.18, 0.01, 0.99)
        
        lc1, lc2 = st.columns(2)
        lc1.html(f'<div class="glass-card"><span class="val-title">Drehmoment</span><br><span class="val-main">{l_torque:.1f} Nm</span></div>')
        lc2.html(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main">{l_temp:.0f} °C</span></div>')
        
        st.html(f"""
            <div class="glass-card" style="border: 1px solid #58a6ff; background: rgba(88,166,255,0.05);">
                <span class="val-title">Prädizierter Zustand: {lab_report['diag']}</span><br>
                <p style="color:#f0f6fc; margin: 8px 0; font-weight:600;">{lab_report['exp']}</p>
                <div style="color:#ff7b72; font-size:1.1rem; font-weight:bold; border-top:1px solid #30363d; padding-top:6px;">{lab_report['act']}</div>
            </div>
        """)

# --- RUNTIME CONTROLS ---
st.divider()
b1, b2 = st.columns(2)
if b1.button("▶ SIMULATION STARTEN / PAUSIEREN", use_container_width=True):
    st.session_state.twin['active'] = not st.session_state.twin['active']
if b2.button("🔄 RESET", use_container_width=True):
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
