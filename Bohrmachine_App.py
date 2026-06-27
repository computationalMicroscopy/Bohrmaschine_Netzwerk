import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="KI-Zerspanungslabor Pro", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #58a6ff;
        margin-bottom: 20px; text-align: center; border-bottom: 2px solid #30363d; padding-bottom: 10px;
    }
    .glass-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 15px; margin-bottom: 10px; text-align: center;
    }
    .val-title { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; font-weight: 600; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #f0f6fc; }
    
    .xai-container { height: 580px; overflow-y: auto; padding-right: 5px; }
    .xai-card {
        background: #1f242c; border-left: 4px solid #58a6ff;
        padding: 12px; border-radius: 6px; margin-bottom: 10px;
    }
    .xai-feature-row { display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e; margin-top: 4px;}
    .xai-bar-bg { background: #21262d; height: 5px; width: 100%; border-radius: 2px; margin-bottom: 6px; }
    .xai-bar-fill { background: #58a6ff; height: 5px; border-radius: 2px; }
    .reason-text { color: #f0f6fc; font-size: 0.9rem; margin-top: 5px; font-weight: 600; }
    .sensor-snapshot { font-size: 0.75rem; color: #8b949e; margin-top: 4px; font-family: monospace; border-bottom: 1px solid #30363d; padding-bottom: 4px;}
    .action-text { color: #ff7b72; font-weight: bold; font-size: 0.8rem; margin-top: 6px; border-top: 1px solid #30363d; padding-top: 6px; }
    .diag-badge { background: #388bfd; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }
    
    .emergency-alert {
        background: #da3633; color: white; padding: 12px; border-radius: 6px; 
        font-weight: bold; text-align: center; margin-bottom: 15px; font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">Präzisions-Bohrerlabor & XAI-Zentrum (Calibrated Physics)</div>', unsafe_allow_html=True)

# --- 2. XAI ROOT-CAUSE ENGINE (PHYSIKALISCH LOGISCH) ---
def get_expert_diagnostics(top_reason, current_vals, settings, integrity):
    vc, f, d = settings['vc'], settings['f'], settings['d']
    
    mapping = {
        "Mechanische Torsions-Überlast": {
            "diag": "CRITICAL TORQUE", 
            "exp": f"Das Drehmoment ({current_vals['M']:.1f} Nm) überlastet den Schaftquerschnitt.", 
            "act": f"ABHILFE: Vorschub f auf {f*0.7:.2f} mm/U senken! Das Reduzieren von vc bringt hier physikalisch keine Entlastung."
        },
        "Thermische Gefüge-Erweichung": {
            "diag": "THERMAL OVERLOAD", 
            "exp": f"Schnittkantentemperatur ({current_vals['T']:.0f}°C) liegt im kritischen Bereich für den Werkstoff.", 
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
        "Axiale Schaft-Knickung": {
            "diag": "AXIAL BUCKLED", 
            "exp": f"Die Vorschubkraft ({current_vals['F']:.0f} N) überschreitet die elastische Stabilitätsgrenze.", 
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

# --- 3. INITIALISIERUNG DES STATE-MACHINES & GLOBAL DATA ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'zyklus': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0,
        'seed': np.random.RandomState(1337)
    }

s = st.session_state.twin

MATERIALIEN = {
    "Baustahl (1.0037)": {"kc1.1": 1800, "mc": 0.25, "wear_factor": 0.02, "t_crit": 450}, 
    "Vergütungsstahl (1.7225)": {"kc1.1": 2100, "mc": 0.24, "wear_factor": 0.05, "t_crit": 550}, 
    "Edelstahl (1.4301)": {"kc1.1": 2300, "mc": 0.22, "wear_factor": 0.12, "t_crit": 600}
}

# Fehlerklassen global definiert zur Vermeidung von Scope-Fehlern
LABELS = [
    "Mechanische Torsions-Überlast", 
    "Thermische Gefüge-Erweichung", 
    "Regeneratives Rattern (Resonanz)", 
    "Kühlungs-Abriss (Adhäsion)", 
    "Axiale Schaft-Knickung", 
    "Normaler Standzeit-Abrieb"
]

# --- 4. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Prozessstellgrößen")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    
    vc = st.slider("vc: Schnittgeschwindigkeit [m/min]", 30, 350, 100)
    f = st.slider("f: Vorschub [mm/U]", 0.05, 0.60, 0.15)
    d = st.slider("d: Werkzeugdurchmesser [mm]", 5.0, 32.0, 12.0)
    kuehlung = st.toggle("Kühlschmierstoff (KSS) aktiv", value=True)
    
    st.divider()
    st.header("🎛️ Sensor-Rauschen & Gain")
    noise_level = st.slider("Signalrauschen (Vibration)", 0.1, 2.0, 0.5)
    
    st.divider()
    schrittweite = st.number_input("Zyklen pro Rechenschritt", 5, 100, 20)
    taktzeit = st.select_slider("Taktung (ms)", options=[500, 200, 100, 0], value=100)

# --- 5. DETILLIERTE PHYSIK-ENGINE (LIVE-PROZESS) ---
if s['active'] and not s['broken'] and not s['stall']:
    s['zyklus'] += schrittweite
    
    # 5.1 Kinetik & Drehzahl
    n = (vc * 1000) / (np.pi * d)
    
    # 5.2 Kienzle-Berechnung
    h = (f / 2.0)
    kc = m['kc1.1'] * (h ** -m['mc'])
    
    s['drehmoment'] = (f * (d**2) * kc) / 8000.0  
    s['vorschubkraft'] = (0.5 * d * f * kc) * 1.3  
    s['leistung'] = (s['drehmoment'] * n) / 9550.0  
    
    # 5.3 Durchmesser-abhängige kritische Belastungsgrenzen
    crit_torque = 0.12 * (d ** 3) 
    crit_force = 320 * (d ** 2)
    
    # 5.4 Thermodynamik (PT1-Glied)
    p_friction_watts = s['drehmoment'] * (n * 2 * np.pi / 60.0)
    kss_factor = 25.0 if kuehlung else 2.5
    t_target = 22.0 + (p_friction_watts * 0.02) / kss_factor
    s['thermik'] += (t_target - s['thermik']) * 0.15 
    
    # 5.5 Schwingungs-Modell (Regeneratives Rattern)
    chatter_trigger = 1.0 + (s['drehmoment'] * 0.1) if (int(n) % 400 < 60) else 0.2
    s['vibration'] = max(0.1, chatter_trigger + s['seed'].normal(0, 0.1) * noise_level)
    
    # 5.6 Feature-Normalisierung für die XAI-Engine
    norm_torque = s['drehmoment'] / crit_torque
    norm_force = s['vorschubkraft'] / crit_force
    norm_temp = s['thermik'] / m['t_crit']
    norm_vibr = s['vibration'] / 8.0
    kss_loss = 1.0 if not kuehlung else 0.0
    norm_wear = (100.0 - s['integritaet']) / 100.0
    
    # 5.7 Mathematisch fundiertes Bayes/Klassifikator-Netzwerk
    weights = [3.0, 3.5, 2.5, 4.0, 3.0, 3.5] # Verschleiß-Gewichtung erhöht für stärkere XAI-Präsenz
    scores = np.array([
        norm_torque * weights[0],
        norm_temp * weights[1],
        norm_vibr * weights[2],
        kss_loss * weights[3],
        norm_force * weights[4],
        norm_wear * weights[5]
    ])
    
    exp_s = np.exp(scores - np.max(scores))
    probabilities = (exp_s / exp_s.sum()) * 100
    evidenz_list = sorted(zip(LABELS, probabilities), key=lambda x: x[1], reverse=True)
    
    # --- DIDAKTISCH KORRIGIERTE RISIKO-KOPPLUNG ---
    max_stress = max([norm_torque, norm_force, norm_temp, norm_vibr])
    # Progressiver Risikoaufschlag durch akkumulierten Verschleiß
    combined_risk_score = max_stress + (norm_wear ** 2.0) * 0.95
    
    if combined_risk_score < 0.75:
        s['risk'] = combined_risk_score * 0.20  
    else:
        s['risk'] = 0.15 + (combined_risk_score - 0.75) * 2.5
    s['risk'] = np.clip(s['risk'], 0.01, 0.99)
    
    # 5.8 Degradation (Verschleißfortschritt)
    v_factor = (vc / 120.0) ** 1.5
    thermal_accelerator = np.exp(max(0.0, s['thermik'] - m['t_crit']) / 30.0)
    s['abrasion'] += (m['wear_factor'] * v_factor * f * thermal_accelerator) * (schrittweite / 10.0)
    
    fatigue = 0.0
    if norm_torque > 0.85: fatigue += (norm_torque - 0.85) ** 2
    if norm_force > 0.85: fatigue += (norm_force - 0.85) ** 2
    if norm_vibr > 0.85: fatigue += (norm_vibr - 0.85) * 0.1
    
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

# --- 6. METRIC DASHBOARD ---
if s['broken']:
    st.markdown('<div class="emergency-alert">💥 STRUKTURELLER WERKZEUGBRUCH! Schneide abgescheert oder Schaft geknickt.</div>', unsafe_allow_html=True)
if s['stall']:
    st.markdown('<div class="emergency-alert">⚠️ SPINDELLAST-STALL: Leistungsaufnahme überschreitet maximales Motordrehmoment (7.5 kW).</div>', unsafe_allow_html=True)

c0, c1, c2, c3, c4, c5, c6 = st.columns(7)
c0.markdown(f'<div class="glass-card"><span class="val-title">Zyklen</span><br><span class="val-main">{s["zyklus"]}</span></div>', unsafe_allow_html=True)
c1.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#2ea44f">{s["integritaet"]:.1f}%</span></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#f85149">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:#ff7b72">{s["thermik"]:.0f}°C</span></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="glass-card"><span class="val-title">Vibration</span><br><span class="val-main" style="color:#a371f7">{s["vibration"]:.2f} mm/s</span></div>', unsafe_allow_html=True)
c5.markdown(f'<div class="glass-card"><span class="val-title">Drehmoment</span><br><span class="val-main" style="color:#e3b341">{s["drehmoment"]:.1f} Nm</span></div>', unsafe_allow_html=True)
c6.markdown(f'<div class="glass-card"><span class="val-title">Leistung</span><br><span class="val-main" style="color:#58a6ff">{s["leistung"]:.2f} kW</span></div>', unsafe_allow_html=True)

# --- 7. TABS: ECHTZEIT VS WAS-WÄRE-WENN LABOR ---
t1, t2 = st.tabs(["📈 Echtzeit-Zustand & Trends", "🔬 Interaktives Was-Wäre-Wenn Labor"])

with t1:
    col_graph, col_log = st.columns([2, 1])
    with col_graph:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                subplot_titles=("Strukturelle Integrität (%)", "Kinetische Lasten: Drehmoment (Nm) & Axialkraft (N)", "Prozessdynamik: Vibration (mm/s) & Temperatur (°C)"))
            
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#2ea44f', width=2.5)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['m'], line=dict(color='#e3b341'), name="Moment"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['f']/100.0, line=dict(color='#1f6feb', dash='dash'), name="Axialkraft/100"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#a371f7'), name="Vibration"), 3, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#ff7b72'), name="Temp"), 3, 1)
            
            fig.update_layout(height=580, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Simulation starten, um Telemetriedaten aufzuzeichnen.")
            
    with col_log:
        st.markdown("### 👁️ XAI: Ursachen-Diagnose")
        if s['logs']:
            html_str = '<div class="xai-container">'
            for l in s['logs'][:10]:
                bars = "".join([f'<div class="xai-feature-row"><span>{e[0]}</span><span>{e[1]:.1f}%</span></div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{e[1]}%"></div></div>' for e in l['evidenz'][:3]])
                html_str += f"""
                <div class="xai-card">
                    <div style="display:flex; justify-content:between; align-items:center; margin-bottom:6px;">
                        <span class="diag-badge">{l['info']['diag']}</span>
                        <span style="font-size:10px; color:#8b949e; margin-left:auto;">{l['zeit']}</span>
                    </div>
                    <div class="reason-text">{l['info']['exp']}</div>
                    <div class="sensor-snapshot">{l['info']['snapshot']}</div>
                    <div style="margin-top:6px;">{bars}</div>
                    <div class="action-text">{l['info']['act']}</div>
                </div>"""
            html_str += '</div>'
            st.markdown(html_str, unsafe_allow_html=True)

# --- RESTAURIERTES UND OPTIMIERTES WAS-WÄRE-WENN LABOR ---
with t2:
    st.markdown("### 🧪 Was-Wäre-Wenn Szenarien-Sandbox")
    st.markdown("Verändere die Parameter völlig autark vom echten Prozess, um zu prüfen, wie die KI in Grenzsituationen oder bei simuliertem Werkzeugverschleiß entscheiden würde.")
    
    col_inputs, col_outputs = st.columns([1, 1])
    
    with col_inputs:
        st.subheader("Hypothetische Stellgrößen")
        lab_mat = st.selectbox("Labor-Werkstoff", list(MATERIALIEN.keys()), key="lab_mat")
        lm = MATERIALIEN[lab_mat]
        
        lab_vc = st.slider("Labor vc [m/min]", 30, 350, 100, key="lab_vc")
        lab_f = st.slider("Labor f [mm/U]", 0.05, 0.60, 0.15, key="lab_f")
        lab_d = st.slider("Labor d [mm]", 5.0, 32.0, 12.0, key="lab_d")
        lab_kss = st.toggle("Labor: KSS aktiv", value=True, key="lab_kss")
        
        st.divider()
        st.subheader("Simulierter Werkzeugzustand")
        lab_integ = st.slider("Virtuelle Integrität des Bohrers (%)", 0.0, 100.0, 100.0, key="lab_integ")
        lab_vibr_override = st.slider("Manuelle Schwingungsüberlagerung [mm/s]", 0.1, 15.0, 0.4, key="lab_vibr")

    with col_outputs:
        st.subheader("Physikalische Labor-Berechnung")
        
        # Exakte Spiegelung der kalibrierten Physik-Engine für Steady-State
        l_n = (lab_vc * 1000) / (np.pi * lab_d)
        l_h = (lab_f / 2.0)
        l_kc = lm['kc1.1'] * (l_h ** -lm['mc'])
        
        l_torque = (lab_f * (lab_d**2) * l_kc) / 8000.0
        l_force = (0.5 * lab_d * lab_f * l_kc) * 1.3
        l_power = (l_torque * l_n) / 9550.0
        
        l_p_friction = l_torque * (l_n * 2 * np.pi / 60.0)
        l_kss_fac = 25.0 if lab_kss else 2.5
        l_temp = 22.0 + (l_p_friction * 0.02) / l_kss_fac
        
        c_t = 0.12 * (lab_d ** 3)
        c_f = 320 * (lab_d ** 2)
        
        # Normalisierungen
        l_norm_torque = l_torque / c_t
        l_norm_force = l_force / c_f
        l_norm_temp = l_temp / lm['t_crit']
        l_norm_vibr = lab_vibr_override / 8.0
        l_kss_loss = 1.0 if not lab_kss else 0.0
        l_norm_wear = (100.0 - lab_integ) / 100.0
        
        # KI-Scores
        l_weights = [3.0, 3.5, 2.5, 4.0, 3.0, 3.5]
        l_scores = np.array([
            l_norm_torque * l_weights[0], l_norm_temp * l_weights[1], l_norm_vibr * l_weights[2],
            l_kss_loss * l_weights[3], l_norm_force * l_weights[4], l_norm_wear * l_weights[5]
        ])
        
        l_exp = np.exp(l_scores - np.max(l_scores))
        l_probs = (l_exp / l_exp.sum()) * 100
        l_evidenz = sorted(zip(LABELS, l_probs), key=lambda x: x[1], reverse=True)
        
        # Risiko-Kopplung
        l_max_stress = max([l_norm_torque, l_norm_force, l_norm_temp, l_norm_vibr])
        l_combined_score = l_max_stress + (l_norm_wear ** 2.0) * 0.95
        
        if l_combined_score < 0.75:
            lab_risk = l_combined_score * 0.20
        else:
            lab_risk = 0.15 + (l_combined_score - 0.75) * 2.5
        lab_risk = np.clip(lab_risk, 0.01, 0.99)
        
        # Visualisierung im Laborraum
        lc1, lc2 = st.columns(2)
        lc1.metric("Errechnetes Moment", f"{l_torque:.1f} Nm", delta=f"Limit: {c_t:.1f} Nm", delta_color="inverse")
        lc2.metric("Errechnete Temperatur", f"{l_temp:.0f} °C", delta=f"Kritisch: {lm['t_crit']} °C", delta_color="inverse")
        
        lc3, lc4 = st.columns(2)
        lc3.metric("Spindellast", f"{l_power:.2f} kW", delta="Max: 7.5 kW", delta_color="inverse")
        lc4.metric("Vorschubkraft", f"{l_force:.0f} N", delta=f"Limit: {c_f:.0f} N", delta_color="inverse")
        
        st.markdown(f"""
            <div class="glass-card" style="border: 2px solid #58a6ff; margin-top:15px; background: #1f242c;">
                <span class="val-title">KI-Prognose Bruchrisiko:</span><br>
                <span class="val-main" style="color:#ff7b72; font-size:2rem;">{lab_risk:.1%}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔬 Prädizierte KI-Urachengewichtung:")
        html_bars = ""
        for name, prob in l_evidenz[:4]:
            html_bars += f"""
            <div class="xai-feature-row" style="margin-top:8px;"><span>{name}</span><span>{prob:.1f}%</span></div>
            <div class="xai-bar-bg" style="height:8px;"><div class="xai-bar-fill" style="width:{prob}%; height:8px;"></div></div>
            """
        st.markdown(html_bars, unsafe_allow_html=True)

# --- 8. SIMULATION RUNTIME CONTROLS ---
st.divider()
b1, b2 = st.columns(2)
if b1.button("▶ SIMULATION START / PAUSE", use_container_width=True):
    s['active'] = not s['active']
if b2.button("🔄 NEUEN BOHRER EINSPANNEN", use_container_width=True):
    st.session_state.twin = {
        'zyklus': 0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'stall': False,
        'thermik': 22.0, 'vibration': 0.2, 'integritaet': 100.0, 'risk': 0.0,
        'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0, 'abrasion': 0.0,
        'seed': np.random.RandomState(1337)
    }
    st.rerun()

if s['active']:
    if taktzeit > 0: time.sleep(taktzeit / 1000)
    st.rerun()
