import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP & STYLING ---
st.set_page_config(layout="wide", page_title="KI-Labor Bohrertechnik", page_icon="⚙️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #e1e4e8; }
    .main-title {
        font-size: 2.5rem; font-weight: 800; color: #e3b341;
        margin-bottom: 20px; text-align: center; border-bottom: 2px solid #e3b341; padding-bottom: 10px;
    }
    .glass-card {
        background: rgba(23, 28, 36, 0.7); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; margin-bottom: 15px;
    }
    .val-title { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.2px; }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 800; }
    
    .xai-container { height: 650px; overflow-y: auto; padding-right: 10px; }
    .xai-card {
        background: rgba(30, 35, 45, 0.9); border-left: 5px solid #e3b341;
        padding: 15px; border-radius: 8px; margin-bottom: 12px;
        font-family: 'Segoe UI', sans-serif;
    }
    .xai-feature-row { display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e; }
    .xai-bar-bg { background: #1b1f23; height: 6px; width: 100%; border-radius: 3px; margin: 4px 0 8px 0; }
    .xai-bar-fill { background: linear-gradient(90deg, #e3b341, #f85149); height: 6px; border-radius: 3px; }
    .reason-text { color: #ffffff; font-size: 0.95rem; margin-top: 10px; font-weight: 600; text-transform: uppercase; }
    .sensor-snapshot { font-size: 0.75rem; color: #3fb950; margin-top: 5px; font-family: monospace; border-bottom: 1px solid #30363d; padding-bottom: 5px;}
    .maint-block { margin-top: 10px; padding: 8px; background: rgba(88, 166, 255, 0.05); border-radius: 4px; }
    .maint-title { font-size: 0.7rem; color: #58a6ff; font-weight: bold; text-transform: uppercase; }
    .maint-text { color: #c9d1d9; font-size: 0.82rem; line-height: 1.4; margin-bottom: 5px;}
    .action-text { color: #f85149; font-weight: bold; font-size: 0.85rem; margin-top: 8px; border-top: 1px solid #30363d; padding-top: 8px; }
    .diag-badge { background: #e3b341; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 900; }
    
    .emergency-alert {
        background: #f85149; color: white; padding: 15px; border-radius: 8px; 
        font-weight: bold; text-align: center; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">KI-Labor Bohrertechnik (Physics-Informed Twin)</div>', unsafe_allow_html=True)

# --- 2. DYNAMISCHE DIAGNOSE-ENGINE (ROOT-CAUSE TARGETED) ---
def get_dynamic_expert_analysis(top_reason, current_vals, settings, integrity):
    vc, f, d, k = settings['vc'], settings['f'], settings['d'], settings['k']
    
    # Physikalisch logische Zuweisung der Abhilfemaßnahmen
    mapping = {
        "Mechanische Überlast (Torsion)": {
            "diag": "DIAGNOSE: DREHMOMENT-EXZESS", 
            "exp": f"Das Schnittmoment ({current_vals['d']:.1f} Nm) überlastet die Torsionsfestigkeit des Schafts.", 
            "maint": f"Spindellast prüfen. Mechanische Integrität liegt bei {integrity:.1f}%.", 
            "act": f"DROSSLUNG: Vorschub f auf {f*0.65:.2f} mm/U senken! (Schnittgeschwindigkeit vc hat kaum Einfluss auf Drehmoment)."
        },
        "Thermische Gefüge-Überhitzung": {
            "diag": "DIAGNOSE: THERMISCHE INSTABILITÄT", 
            "exp": f"Prozesstemperatur ({current_vals['t']:.0f}°C) erweicht die Schneidkante durch massive Reibungsleistung.", 
            "maint": "Gefahr von Kolkverschleiß und plastischer Deformation der Hauptschneide.", 
            "act": f"TEMPERATURSENKUNG: Schnittgeschwindigkeit vc auf {int(vc*0.75)} m/min reduzieren oder Kühlmitteldruck erhöhen."
        },
        "Resonanz-Rattern": {
            "diag": "DIAGNOSE: DYNAMISCHES CHATTER", 
            "exp": f"Vibrationsamplitude ({current_vals['v']:.1f} mm/s) deutet auf regeneratives Rattern hin.", 
            "maint": "Erhöhter Freiflächenverschleiß durch stoßartige Belastung der Schneide.", 
            "act": f"FREQUENZ-SHIFT: Drehzahl anpassen. vc testweise um 15% verändern (auf {int(vc*1.15)} m/min oder {int(vc*0.85)} m/min), um Resonanz zu brechen."
        },
        "Kühlungs-Abriss (Tribologie)": {
            "diag": "DIAGNOSE: KÜHLMITTEL-AUSFALL", 
            "exp": "Akutes Tribologie-Versagen. Spanflächenreibung steigt mangels Schmierung exponentiell an.", 
            "maint": "Gefahr von thermischen Schockrissen im Hartmetallgefüge.", 
            "act": "NOT-AUS EMPFOHLEN: Vorschub stoppen, Spülung und KSS-Zuleitung prüfen."
        },
        "Kritische Schaft-Knickung": {
            "diag": "DIAGNOSE: AXIALKRAFT-EXZESS", 
            "exp": f"Die Vorschubkraft ({current_vals['f_f']:.0f} N) nähert sich der elastischen Knickgrenze nach Euler.", 
            "maint": "Führung und Achsausrichtung der Werkzeugspindel kalibrieren.", 
            "act": f"VORSCHUB-REDUKTION: f sofort auf {f*0.5:.2f} mm/U halbieren, um Achslast zu mindern!"
        },
        "Abrasiver Standzeit-Verschleiß": {
            "diag": "DIAGNOSE: GEFÜGESCHADEN / WEAR", 
            "exp": f"Werkzeug-Integrität kritisch ({integrity:.1f}%). Normaler abrasiver Verschleiß verbraucht.", 
            "maint": "Schneidkanten-Geometrie stark abgeflacht. Erhöhte Leistungsaufnahme.", 
            "act": "WERKZEUGWECHSEL: Zyklus beenden, Bohrer im Revolver gegen Neuwerkzeug tauschen."
        }
    }
    res = mapping.get(top_reason, {"diag": "DIAGNOSE: NOMINAL", "exp": "Prozess läuft innerhalb des physikalischen Fensters.", "maint": "Routine-Überwachung.", "act": "Keine Korrektur erforderlich."})
    res["snapshot"] = f"IST: {current_vals['t']:.0f}°C | {current_vals['v']:.1f} mm/s | {current_vals['d']:.1f} Nm | {current_vals['p']:.2f} kW"
    return res

def calculate_metrics_bayesian(prior_risk, norm_torque, norm_power, norm_temp, norm_vibr, norm_force, kuehl_ausfall, integrity):
    # Physikalische Gewichtungsmatrix für den Klassifikator
    # Zeilen-Indizes der Fehlerklassen korrespondieren mit den dominanten physikalischen Stimuli
    w = [2.8, 3.5, 3.0, 4.5, 3.2, 2.0]
    
    raw_scores = np.array([
        norm_torque * w[0],                  # Mechanische Überlast (Torsion)
        norm_temp * w[1],                    # Thermische Überhitzen
        norm_vibr * w[2],                    # Resonanz-Rattern
        kuehl_ausfall * w[3],                # Kühlungs-Abriss
        norm_force * w[4],                   # Schaft-Knickung
        ((100.0 - integrity) / 100.0) * w[5] # Abrasiver Verschleiß
    ])
    
    # Numerisch stabiler Softmax für klare, gleitende Evidenzbalken (XAI)
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    probabilities = (exp_scores / exp_scores.sum()) * 100
    
    # Probabilistisches Gesamtrisiko über logistischen Aktivierungs-Kanal
    z = (norm_torque * 1.5) + (norm_temp * 2.0) + (norm_vibr * 1.2) + (kuehl_ausfall * 2.5) + (norm_force * 1.5) + ((100-integrity)/50.0)
    likelihood = 1.0 / (1.0 + np.exp(-(z - 4.0)))
    posterior = (likelihood * 0.5) + (prior_risk * 0.5)
    
    labels = ["Mechanische Überlast (Torsion)", "Thermische Gefüge-Überhitzung", "Resonanz-Rattern", "Kühlungs-Abriss (Tribologie)", "Kritische Schaft-Knickung", "Abrasiver Standzeit-Verschleiß"]
    evidenz = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
    
    # Restlebensdauer (RUL) basierend auf der Schadensakkumulationsrate
    rul = int(max(0, (integrity - 2) / max(0.01, (posterior * 0.6))) * 6) if posterior < 0.95 else 0
    return np.clip(posterior, 0.001, 0.999), evidenz, rul

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.5, 'risk': 0.01, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 1200, 'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0}

MATERIALIEN = {
    "Baustahl (1.0037)": {"kc1.1": 1900, "mc": 0.26, "rate": 0.12, "t_crit": 480}, 
    "Vergütungsstahl (1.7225)": {"kc1.1": 2200, "mc": 0.24, "rate": 0.22, "t_crit": 560}, 
    "Edelstahl (1.4301)": {"kc1.1": 2450, "mc": 0.21, "rate": 0.48, "t_crit": 620}, 
    "Titan Grade 5 (3.7165)": {"kc1.1": 2950, "mc": 0.23, "rate": 1.15, "t_crit": 720}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Physikalische Parameter")
    mat_name = st.selectbox("Werkstoff (Werkstück)", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("vc: Schnittgeschwindigkeit [m/min]", 20, 450, 120)
    f = st.slider("f: Vorschub pro Umdrehung [mm/U]", 0.02, 0.80, 0.18)
    d = st.number_input("d: Bohrer-Durchmesser [mm]", 3.0, 40.0, 12.0)
    kuehlung = st.toggle("Kühlschmierstoff (KSS) aktiv", value=True)
    
    st.divider()
    st.header("📡 Sensor-Kanal-Verstärkung")
    sens_vibr = st.slider("Vibrations-Gain", 0.5, 3.0, 1.0)
    sens_load = st.slider("Last-Gain", 0.5, 3.0, 1.0)
    
    st.divider()
    zyklus_sprung = st.number_input("Simulations-Schrittweite (Zyklen)", 1, 50, 10)
    sim_takt = st.select_slider("Aktualisierungsintervall (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. DETILLIERTE PHYSIK-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    
    # 5.1 Drehzahl berechnen aus vc und d
    n = (vc * 1000) / (np.pi * d)
    
    # 5.2 Kienzle-Gleichung für spezifische Schnittkraft (Schnitttiefe h = f/2 bei 2 Schneiden)
    f_z = max(0.01, f / 2.0)
    kc = m['kc1.1'] * (f_z ** -m['mc'])
    
    # 5.3 Physikalische Hauptkräfte
    s['drehmoment'] = ((f * (d**2) * kc) / 8000.0) * sens_load
    s['vorschubkraft'] = (0.5 * d * f * kc) * sens_load
    s['leistung'] = (s['drehmoment'] * n) / 9550.0  # Schnittleistung in kW
    
    # 5.4 Thermodynamische Bilanz (Wärmeeintrag vs. KSS-Konvektion)
    p_heat_watts = s['drehmoment'] * (n * 2 * np.pi / 60.0)
    kss_dissipation = 14.0 if kuehlung else 1.8
    t_equilibrium = 22.0 + (p_heat_watts * 0.04) / kss_dissipation
    s['thermik'] += (t_equilibrium - s['thermik']) * 0.20  # Thermische Trägheit (PT1)
    
    # 5.5 Dynamisches Rattern (Vibration) gekoppelt an Schnittkraft und Drehzahlstabilität
    chatter_base = (s['drehmoment'] * 0.05 + (vc / 100.0)) * sens_vibr
    s['vibration'] = chatter_base + s['seed'].normal(1.2, 0.25)
    
    # 5.6 Feature-Normalisierung für das Bayes-Netzwerk
    norm_torque = s['drehmoment'] / 65.0              # Skaliert auf Schaftfestigkeit
    norm_power = s['leistung'] / 5.5                  # Skaliert auf Maschinengrenze 5.5kW
    norm_temp = s['thermik'] / m['t_crit']            # Skaliert auf Werkstoff-Erweichung
    norm_vibr = s['vibration'] / 14.0                 # Skaliert auf Rattergrenze
    norm_force = s['vorschubkraft'] / 6000.0          # Skaliert auf Knickgrenze
    kuehl_ausfall_val = 1.0 if not kuehlung else 0.0
    
    # 5.7 KI-Metriken berechnen
    s['risk'], evidenz_list, s['rul'] = calculate_metrics_bayesian(
        s['risk'], norm_torque, norm_power, norm_temp, norm_vibr, norm_force, kuehl_ausfall_val, s['integritaet']
    )
    
    # 5.8 Strukturelle Schädigungs-Akkumulation (Verschleißfortschritt)
    # Beschleunigter Verschleiß bei kritischer Thermik (Arrhenius-Anteil)
    thermal_wear_acc = np.exp(max(0.0, s['thermik'] - m['t_crit']) / 40.0)
    s['verschleiss'] += ((m['rate'] * (vc**1.4) * f * thermal_wear_acc) / 45000.0) * zyklus_sprung
    
    # Integritätsverlust durch mechanische Überlasten, Schwingbruch und abrasiven Abrieb
    damage_step = ((s['vorschubkraft']/5500.0)**2 + (s['drehmoment']/60.0)**2 + (s['vibration']/12.0)*0.08 + (s['verschleiss']*0.01)) * zyklus_sprung
    s['integritaet'] -= damage_step
    
    if s['integritaet'] <= 0 or s['leistung'] > 6.5: 
        s['broken'], s['active'], s['integritaet'] = True, False, 0
    
    # Log-Generierung
    expert_info = get_dynamic_expert_analysis(evidenz_list[0][0], {'t': s['thermik'], 'v': s['vibration'], 'd': s['drehmoment'], 'p': s['leistung'], 'f_f': s['vorschubkraft']}, {'vc': vc, 'f': f, 'd': d, 'k': kuehlung}, s['integritaet'])
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'info': expert_info, 'evidenz': evidenz_list})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration'], 'p': s['leistung']})

# --- 6. UI BENUTZEROBERFLÄCHE ---
if s['broken']: 
    if s['leistung'] > 6.5:
        st.markdown('<div class="emergency-alert">🚨 MOTOR-STALL: Spindelleistung überschritten (Überlast-Stopp)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="emergency-alert">🚨 SYSTEM-STOPP: GEFÜGEBRUCH ODER KNICKUNG DES WERKZEUGS</div>', unsafe_allow_html=True)

m0, m1, m2, m3, m4, m5, m6 = st.columns(7)
m0.markdown(f'<div class="glass-card"><span class="val-title">Zyklen</span><br><span class="val-main">{s["zyklus"]}</span></div>', unsafe_allow_html=True)
m1.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.1f}%</span></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#f85149">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="glass-card"><span class="val-title">Restzeit (RUL)</span><br><span class="val-main" style="color:#58a6ff">{s["rul"]} Z.</span></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:#f85149">{s["thermik"]:.0f}°C</span></div>', unsafe_allow_html=True)
m5.markdown(f'<div class="glass-card"><span class="val-title">Effektiv-Leistung</span><br><span class="val-main" style="color:#bc8cff">{s["leistung"]:.2f} kW</span></div>', unsafe_allow_html=True)
m6.markdown(f'<div class="glass-card"><span class="val-title">Drehmoment</span><br><span class="val-main">{s["drehmoment"]:.1f} Nm</span></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 LIVE-TELEMETRIE & TWIN", "🧪 PHYSlKALISCHES DIAGNOSE-LABOR"])

with tab1:
    col_l, col_r = st.columns([2.2, 1.8])
    with col_l:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Strukturelle Integrität (%)", "Prozessleistung (kW) & Temperatur (°C)", "Klassifikator: Anomalie-Wahrscheinlichkeit (%)"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#3fb950', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#f85149'), name="Temp"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['p']*100, line=dict(color='#bc8cff'), name="Leistung (skaliert)"), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, line=dict(color='#e3b341', width=3)), 3, 1)
            fig.update_layout(height=650, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown("### 🧠 Deep XAI: Diagnosezentrum")
        xai_html = '<div class="xai-container">'
        for l in s['logs'][:15]:
            features = "".join([f'<div class="xai-feature-row"><span>{e[0]}</span><span>{e[1]:.1f}%</span></div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{e[1]}%"></div></div>' for e in l['evidenz'][:3]])
            xai_html += f"""
            <div class="xai-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="diag-badge">{l['info']['diag']}</span>
                    <b style="font-size:11px; color:#8b949e;">LOG {l['zeit']} | KI-KONFIDENZ: {max([e[1] for e in l['evidenz']]):.1f}%</b>
                </div>
                <div class="reason-text">{l['info']['exp']}</div>
                <div class="sensor-snapshot">{l['info']['snapshot']}</div>
                <div style="margin-top:10px;">{features}</div>
                <div class="maint-block">
                    <div class="maint-title">Gefahrenanalyse & Tribologie:</div>
                    <div class="maint-text">{l['info']['maint']}</div>
                </div>
                <div class="action-text">REAKTIONS-PROTOKOLL: {l['info']['act']}</div>
            </div>"""
        xai_html += '</div>'
        st.markdown(xai_html, unsafe_allow_html=True)

with tab2:
    st.header("🧪 Beanspruchungs-Simulator (Modell-Stresstest)")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        sim_torque = st.slider("Simulierte Torsion [Nm]", 0.0, 100.0, 25.0)
        sim_power = st.slider("Simulierte Spindelleistung [kW]", 0.0, 8.0, 2.0)
        sim_force = st.slider("Simulierte Axialkraft [N]", 0, 8000, 1500)
    with sc2:
        sim_temp = st.slider("Simulierte Kanten-Hitze [°C]", 20, 1000, 120)
        sim_vibr = st.slider("Simulierte Vibration [mm/s]", 0.0, 30.0, 2.0)
        sim_kuehl = st.toggle("Simuliere totalen KSS-Abriss")
    with sc3:
        # Perfekte Abbildung der Live-Normalisierung im Simulationsraum
        r_sim, evidenz_sim, rul_sim = calculate_metrics_bayesian(
            0.5, sim_torque/65.0, sim_power/5.5, sim_temp/550.0, sim_vibr/14.0, sim_force/6000.0, 1.0 if sim_kuehl else 0.0, 100.0
        )
        
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border: 2px solid #e3b341;">
                <span class="val-title">Berechnete Anomalieklasse im Labor</span><br>
                <span class="val-main" style="color:#e3b341">{evidenz_sim[0][0]}</span>
                <p style="font-size: 0.85rem; color: #8b949e; margin-top: 10px;">
                    Klassifikations-Wahrscheinlichkeit: <b>{evidenz_sim[0][1]:.1f}%</b> | Aggregiertes Bruchrisiko: {r_sim:.1%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        fig_radar = go.Figure(data=go.Scatterpolar(r=[min(100, sim_torque/0.65), min(100, sim_power/0.055), min(100, sim_temp/6.0), min(100, sim_vibr/0.14), (100 if sim_kuehl else 0)], theta=['Torsion','Spindelleistung','Temperatur','Vibration','KSS-Ausfall'], fill='toself', line=dict(color='#f85149')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ SIMULATION STARTEN / PAUSIEREN", use_container_width=True): s['active'] = not s['active']
if c2.button("🔄 NEUES WERKZEUG IN REVOLVER LADEN", use_container_width=True):
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.5, 'risk': 0.01, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 1200, 'drehmoment': 0.0, 'leistung': 0.0, 'vorschubkraft': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
