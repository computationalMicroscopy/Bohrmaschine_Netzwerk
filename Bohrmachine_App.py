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
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; }
    
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

    /* --- REALISTISCHER ANIMATIONS-CONTAINER (FLACKERFREI) --- */
    .drill-stage {
        background: #0d1117; border: 1px solid #30363d; border-radius: 12px;
        height: 600px; display: flex; flex-direction: column; justify-content: center; align-items: center;
        overflow: hidden; position: relative; perspective: 1000px;
    }
    .material-block {
        width: 180px; height: 160px; position: absolute; bottom: 80px;
        border-radius: 6px; box-shadow: inset 0 0 20px rgba(0,0,0,0.8), 0 10px 25px rgba(0,0,0,0.5);
        transition: background-color 0.5s ease; z-index: 1;
    }
    .bore-hole {
        position: absolute; top: 0; left: 50%; transform: translateX(-50%);
        width: 40px; background: #05070a; border-radius: 0 0 50px 50px;
        box-shadow: inset 0 5px 15px rgba(0,0,0,1); transition: height 0.1s linear;
    }
    .drill-assembly {
        position: absolute; left: 50%; transform-style: preserve-3d;
        display: flex; flex-direction: column; align-items: center; z-index: 2;
    }
    .drill-shank {
        width: 34px; height: 120px;
        background: linear-gradient(90deg, #4f5d65 0%, #d1d8dc 50%, #4f5d65 100%);
        border-radius: 4px 4px 0 0;
    }
    .drill-helix {
        width: 30px; height: 180px;
        background: repeating-linear-gradient(145deg, #2c3539, #2c3539 15px, #708090 15px, #d1d8dc 30px);
        border-radius: 0 0 15px 15px; position: relative;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">KI-Labor Bohrertechnik</div>', unsafe_allow_html=True)

# --- 2. DYNAMISCHE DIAGNOSE-ENGINE ---
def get_dynamic_expert_analysis(top_reason, current_vals, settings, integritaet):
    vc, f, d, k = settings['vc'], settings['f'], settings['d'], settings['k']
    integ_impact = "KRITISCH" if integritaet < 50 else ("DEGRADED" if integritaet < 80 else "STABIL")
    integ_text = f"STATUS: {integritaet:.1f}% ({integ_impact}). "
    
    if current_vals['d'] > (d * 10) and integritaet > 80:
        integ_detail = "GEVALTBRUCH-WARNUNG: Die mechanische Last überschreitet die Biegebruchfestigkeit trotz hoher Integrität."
    else:
        integ_detail = "Stabile Risszähigkeit." if integritaet > 50 else "Geringe Risszähigkeit verstärkt Lastrisiko."

    mapping = {
        "Material-Ermüdung": {"diag": "DIAGNOSE: ADHÄSIVER VERSCHLEISS", "exp": f"Gefüge-Ermüdung bei vc={vc}. {integ_detail}", "maint": f"Check der Freiflächen. {integ_text}", "act": f"REDUKTION: vc auf {int(vc*0.85)} m/min."},
        "Überlastung": {"diag": "DIAGNOSE: MECHANISCHE ÜBERLAST", "exp": f"Vorschub f={f} erzeugt {current_vals['d']:.1f}Nm. {integ_detail}", "maint": f"Check Aufnahme. {integ_text}", "act": f"KORREKTUR: f auf {f*0.7:.2f}mm/U begrenzen."},
        "Gefüge-Überhitzung": {"diag": "DIAGNOSE: THERMISCHE ÜBERLAST", "exp": f"Temperatur ({current_vals['t']:.0f}°C). {integ_detail}", "maint": f"Check auf Kolkverschleiß. {integ_text}", "act": "KÜHLUNG: vc senken oder Druck erhöhen."},
        "Resonanz-Instabilität": {"diag": "DIAGNOSE: DYNAMISCHE INSTABILITÄT", "exp": f"Vibration {current_vals['v']:.2f}mm/s. {integ_detail}", "maint": f"Auskraglänge prüfen. {integ_text}", "act": f"SHIFT: vc auf {int(vc*0.9)} variieren."},
        "Kühlungs-Defizit": {"diag": "DIAGNOSE: TRIBOLOGIE-VERSAGEN", "exp": f"Schmierfilmabriss. {integ_detail}", "maint": f"Konzentration prüfen. {integ_text}", "act": "SYSTEMCHECK: Kühlung blockiert."},
        "Struktur-Vorschaden": {"diag": "DIAGNOSE: GEFÜGESCHADEN", "exp": f"Integrität {integritaet:.1f}%. {integ_detail}", "maint": f"Emissionsprüfung. {integ_text}", "act": "NOT-AUS: Wechsel einleiten."}
    }
    res = mapping.get(top_reason, {"diag": "DIAGNOSE: STABIL", "exp": "Parameter OK.", "maint": "Routine.", "act": "Keine Korrektur."})
    res["snapshot"] = f"IST: {current_vals['t']:.1f}°C | {current_vals['v']:.2f} mm/s | {current_vals['d']:.1f} Nm"
    return res

def calculate_metrics_bayesian(prior_risk, alter, last, thermik, vibration, kuehlung_ausfall, integritaet):
    w = [1.2, 2.4, 3.8, 4.2, 4.5, 0.10]
    raw_scores = np.array([alter * w[0], last * w[1], thermik * w[2], (vibration/10) * w[3], kuehlung_ausfall * w[4], (100 - integritaet) * w[5]])
    exp_scores = np.exp(raw_scores * 0.8) 
    probabilities = (exp_scores / exp_scores.sum()) * 100
    
    z = sum(raw_scores)
    likelihood = 1 / (1 + np.exp(-(z - 9.5)))
    posterior = (likelihood * 0.3) + (prior_risk * 0.7)
    
    labels = ["Material-Ermüdung", "Überlastung", "Gefüge-Überhitzung", "Resonanz-Instabilität", "Kühlungs-Defizit", "Struktur-Vorschaden"]
    evidenz = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
    
    divisor = max(0.001, (posterior * 0.45))
    rul = int(max(0, (integritaet - 10) / divisor) * 5.5) if posterior < 0.98 else 0
    return np.clip(posterior, 0.001, 0.999), evidenz, rul

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.01, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}

MATERIALIEN = {
    "Baustahl": {"kc1.1": 1900, "mc": 0.26, "rate": 0.15, "t_crit": 450, "color": "linear-gradient(135deg, #5a6975, #3a444a)"}, 
    "Vergütungsstahl": {"kc1.1": 2100, "mc": 0.25, "rate": 0.25, "t_crit": 550, "color": "linear-gradient(135deg, #3b596d, #243743)"}, 
    "Edelstahl": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650, "color": "linear-gradient(135deg, #b0b0b0, #7a7a7a)"}, 
    "Titan Grade 5": {"kc1.1": 2800, "mc": 0.24, "rate": 1.2, "t_crit": 750, "color": "linear-gradient(135deg, #606d7d, #3c444e)"}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfiguration")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]; vc = st.slider("vc [m/min]", 20, 600, 180); f = st.slider("f [mm/U]", 0.01, 1.2, 0.2); d = st.number_input("Ø [mm]", 1.0, 100.0, 12.0); kuehlung = st.toggle("Kühlung aktiv", value=True)
    st.divider(); st.header("📡 Sensoren")
    sens_vibr = st.slider("Vibrations-Gain (mm/s)", 0.1, 5.0, 1.0); sens_load = st.slider("Last-Gain (Nm)", 0.1, 5.0, 1.0)
    st.divider(); zyklus_sprung = st.number_input("Schrittweite", 1, 100, 10); sim_takt = st.select_slider("Takt (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. PHYSIK-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    s['drehmoment'] = ((m['kc1.1'] * (f ** -m['mc']) * f * (d/2)**2) / 1000) * sens_load
    s['verschleiss'] += ((m['rate'] * (vc**1.7) * f) / (12000 if kuehlung else 300)) * zyklus_sprung
    s['thermik'] += ((22 + (s['verschleiss']*1.4) + (vc*0.22) + (0 if kuehlung else 280)) - s['thermik']) * 0.25
    s['vibration'] = ((s['verschleiss']*0.04 + vc*0.005 + s['drehmoment']*0.02) * sens_vibr) + s['seed'].normal(1.5, 0.3)
    
    s['risk'], evidenz_list, s['rul'] = calculate_metrics_bayesian(s['risk'], s['zyklus']/1000, s['drehmoment']/60, s['thermik']/m['t_crit'], s['vibration'], 1.0 if not kuehlung else 0.0, s['integritaet'])
    s['integritaet'] -= ((s['verschleiss']/100)*0.04 + (s['drehmoment']/100)*0.01 + (np.exp(max(0, s['thermik']-m['t_crit'])/45)-1)*2 + (max(0,s['vibration'])/25)*0.05) * zyklus_sprung
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0
    
    expert_info = get_dynamic_expert_analysis(evidenz_list[0][0], {'t': s['thermik'], 'v': s['vibration'], 'd': s['drehmoment']}, {'vc': vc, 'f': f, 'd': d, 'k': kuehlung}, s['integritaet'])
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'info': expert_info, 'evidenz': evidenz_list})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration']})

# --- 6. UI ---
if s['broken']: st.markdown('<div class="emergency-alert">🚨 SYSTEM-STOPP: WERKZEUGBRUCH</div>', unsafe_allow_html=True)

m0, m1, m2, m3, m4, m5, m6 = st.columns(7)
m0.markdown(f'<div class="glass-card"><span class="val-title">Vergangene Zyklen</span><br><span class="val-main">{s["zyklus"]}</span></div>', unsafe_allow_html=True)
m1.markdown(f'<div class="glass-card"><span class="val-title">Integrität</span><br><span class="val-main" style="color:#3fb950">{s["integritaet"]:.1f}%</span></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="glass-card"><span class="val-title">Bruchrisiko</span><br><span class="val-main" style="color:#e3b341">{s["risk"]:.1%}</span></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="glass-card"><span class="val-title">Wartung in</span><br><span class="val-main" style="color:#58a6ff">{s["rul"]} Z.</span></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="glass-card"><span class="val-title">Temperatur</span><br><span class="val-main" style="color:#f85149">{s["thermik"]:.0f}°C</span></div>', unsafe_allow_html=True)
m5.markdown(f'<div class="glass-card"><span class="val-title">Vibration (mm/s)</span><br><span class="val-main" style="color:#bc8cff">{max(0,s["vibration"]):.1f}</span></div>', unsafe_allow_html=True)
m6.markdown(f'<div class="glass-card"><span class="val-title">Last (Nm)</span><br><span class="val-main">{s["drehmoment"]:.1f}</span></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 LIVE-ANALYSE", "🧪 SZENARIO-LABOR"])

with tab1:
    col_l, col_m, col_r = st.columns([1.8, 2.2, 1.8])
    
    with col_l:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Historie: Integrität", "Sensorik: Temperatur & Vibration", "KI: Bruchrisiko %"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], fill='tozeroy', line=dict(color='#3fb950', width=3)), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['t'], line=dict(color='#f85149')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['v'], line=dict(color='#bc8cff')), 2, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, line=dict(color='#e3b341', width=3)), 3, 1)
            fig.update_layout(height=650, template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
    with col_m:
        st.markdown("### 🌀 Hardwarebeschleunigter Digitaler Zwilling")
        
        # Echtzeit-Berechnungen für CSS-Steuerung
        hole_depth = min(150, int((s['zyklus'] / 250) * 12))  # Bohrtiefe bis 150px
        drill_y_offset = -60 + hole_depth                    # Vorschubbewegung nach unten
        
        # Dynamische Rotationsgeschwindigkeit gekoppelt an vc
        rot_speed = max(0.0, (650 - vc) / 110) if s['active'] else 0
        anim_rotation = f"animation: spin {rot_speed}s linear infinite;" if rot_speed > 0 else ""
        
        # Thermisches Glühen (Spitzen-Farbüberlagerung)
        temp_ratio = min(1.0, max(0.0, (s['thermik'] - 100) / (m['t_crit'] - 100)))
        glow_color = f"rgba(248, 81, 73, {temp_ratio * 0.85})"
        
        # Vibrations-Amplitude als CSS-Shaking-Intensität
        v_intensity = min(6, int(max(0, s['vibration']) * 0.4)) if s['active'] else 0
        anim_shake = f"animation: shake {0.1}s infinite alternate;" if v_intensity > 0 else ""

        # Werkzeug-Zustand steuern (Bruch)
        broken_css = "transform: translateY(40px) rotate(12deg); opacity: 0.8; filter: grayscale(1) brightness(0.4);" if s['broken'] else ""
        
        # Render schlüsselfertiges flackerfreies HTML5/CSS3 Interface
        st.markdown(f"""
            <div class="drill-stage">
                <div class="drill-assembly" style="top: {drill_y_offset}px; transform: translateX(-50%); {anim_shake} {broken_css}">
                    <div class="drill-shank"></div>
                    <div class="drill-helix" style="{anim_rotation}">
                        <div style="position: absolute; bottom: 0; left:0; width: 100%; height: 60px; 
                                    background: linear-gradient(to top, {glow_color}, transparent); 
                                    border-radius: 0 0 15px 15px; pointer-events: none;"></div>
                    </div>
                </div>
                
                <div class="material-block" style="background: {m['color']};">
                    <div class="bore-hole" style="height: {hole_depth}px;"></div>
                </div>
                
                <style>
                    @keyframes spin {{ 100% {{ transform: rotate(-360deg); }} }}
                    @keyframes shake {{
                        0% {{ margin-left: -{v_intensity}px; margin-top: 0px; }}
                        100% {{ margin-left: {v_intensity}px; margin-top: {v_intensity//2}px; }}
                    }}
                </style>
            </div>
        """, unsafe_allow_html=True)
        
    with col_r:
        st.markdown("### 🧠 Deep XAI: Diagnosezentrum")
        xai_html = '<div class="xai-container">'
        for l in s['logs'][:15]:
            features = "".join([f'<div class="xai-feature-row"><span>{e[0]}</span><span>{e[1]:.1f}%</span></div><div class="xai-bar-bg"><div class="xai-bar-fill" style="width:{e[1]}%"></div></div>' for e in l['evidenz'][:3]])
            xai_html += f"""
            <div class="xai-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="diag-badge">{l['info']['diag']}</span>
                    <b style="font-size:11px; color:#8b949e;">LOG {l['zeit']} | KI-SICHERHEIT: {max([e[1] for e in l['evidenz']]):.1f}%</b>
                </div>
                <div class="reason-text">{l['info']['exp']}</div>
                <div class="sensor-snapshot">{l['info']['snapshot']}</div>
                <div style="margin-top:10px;">{features}</div>
                <div class="maint-block">
                    <div class="maint-title">Prüfprotokoll & Instandhaltung:</div>
                    <div class="maint-text">{l['info']['maint']}</div>
                </div>
                <div class="action-text">HANDLUNGSANWEISUNG: {l['info']['act']}</div>
            </div>"""
        xai_html += '</div>'
        st.markdown(xai_html, unsafe_allow_html=True)

with tab2:
    st.header("🧪 Was-Wäre-Wenn Labor")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        sim_alter = st.slider("Sim. Alter [Zyklen]", 0, 3000, 500)
        sim_last = st.slider("Sim. Last [Nm]", 0, 300, 40)
        sim_vibr = st.slider("Sim. Vibration [mm/s]", 0.0, 50.0, 5.0)
    with sc2:
        sim_temp = st.slider("Sim. Temp. [°C]", 20, 1200, 150)
        sim_integ = st.slider("Integrität [%]", 0, 100, 100)
        sim_kuehl = st.toggle("Sim. Kühlungs-Ausfall")
    with sc3:
        r_sim, evidenz_sim, rul_sim = calculate_metrics_bayesian(0.5, sim_alter/800, sim_last/50, sim_temp/500, sim_vibr, 1.0 if sim_kuehl else 0.0, sim_integ)
        
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border: 2px solid #58a6ff;">
                <span class="val-title">Voraussichtliche Restzyklen</span><br>
                <span class="val-main" style="color:#58a6ff">{rul_sim} Zyklen</span>
                <p style="font-size: 0.8rem; color: #8b949e; margin-top: 10px;">
                    Berechnet auf Basis von Risiko-Posterior: {r_sim:.1%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        fig_radar = go.Figure(data=go.Scatterpolar(r=[sim_alter/30, sim_last/3, sim_temp/12, sim_vibr*2, (100 if sim_kuehl else 0)], theta=['Alter','Last','Hitze','Vibration','Kühlung'], fill='toself', line=dict(color='#e3b341')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=False, height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
if c1.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
if c2.button("🔄 NEUES WERKZEUG", use_container_width=True):
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.01, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
