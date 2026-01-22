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
        margin-bottom: 20px; text-align: center; border-bottom: 2px solid #e3b341;
        padding-bottom: 10px;
    }
    .glass-card {
        background: rgba(23, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px; margin-bottom: 15px;
    }
    .val-main { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 800; }
    .xai-card {
        background: rgba(30, 35, 45, 0.9);
        border-left: 5px solid #e3b341;
        padding: 15px; border-radius: 8px; margin-bottom: 15px;
    }
    .reason-text { color: #e1e4e8; font-size: 0.9rem; margin-top: 8px; line-height: 1.4; }
    .action-text { color: #58a6ff; font-weight: bold; font-size: 0.85rem; margin-top: 5px; }
    .xai-label { color: #8b949e; font-size: 0.75rem; }
    .xai-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #f85149; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">KI-Labor Bohrertechnik</div>', unsafe_allow_html=True)

# --- 2. KI-LOGIK MIT DIDAKTISCHER ERKLÄRUNG ---
def get_didactic_explanation(evidenz_list):
    top_reason = evidenz_list[0][0]
    
    mapping = {
        "Material-Ermüdung": ("Die KI erkennt Mikrorisse im Werkzeugstahl.", "Tausche das Werkzeug demnächst aus, bevor die Risse die Bohrung ungenau machen."),
        "Überlastung": ("Das Drehmoment ist zu hoch für diesen Durchmesser.", "Reduziere den Vorschub (f), um den Bohrer mechanisch zu entlasten."),
        "Gefüge-Überhitzung": ("Die Hitze erreicht die Anlasstemperatur des Stahls – er wird weich.", "Prüfe die Kühlung oder reduziere die Schnittgeschwindigkeit (vc)."),
        "Resonanz-Instabilität": ("Starke Schwingungen 'hämmern' gegen die Schneidkante.", "Drehzahl leicht verändern, um aus dem Resonanzbereich zu kommen."),
        "Kühlungs-Defizit": ("Fehlendes KSS führt zu extremer Reibung in der Spannut.", "Sofort KSS prüfen! Die Späne könnten sonst verschweißen."),
        "Struktur-Vorschaden": ("Ein alter Schaden schwächt die gesamte Stabilität.", "Besondere Vorsicht: Das Werkzeug kann jederzeit schlagartig brechen.")
    }
    return mapping.get(top_reason, ("Stabiler Prozess.", "Keine Korrektur nötig."))

def calculate_metrics(alter, last, thermik, vibration, kss_ausfall, integritaet):
    w = [1.2, 2.4, 3.8, 3.0, 4.5, 0.10]
    scores = [alter * w[0], last * w[1], thermik * w[2], vibration * w[3], kss_ausfall * w[4], (100 - integritaet) * w[5]]
    z = sum(scores)
    risk = 1 / (1 + np.exp(-(z - 9.5)))
    
    labels = ["Material-Ermüdung", "Überlastung", "Gefüge-Überhitzung", "Resonanz-Instabilität", "Kühlungs-Defizit", "Struktur-Vorschaden"]
    evidenz = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    
    if risk > 0.98 or integritaet <= 0: rul = 0
    else: rul = int(max(0, (integritaet - 10) / max(0.01, (risk * 0.45))) * 5.5)
        
    return np.clip(risk, 0.001, 0.999), evidenz, rul

# --- 3. INITIALISIERUNG ---
if 'twin' not in st.session_state:
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}

MATERIALIEN = {
    "Baustahl": {"kc1.1": 1900, "mc": 0.26, "rate": 0.15, "t_crit": 450},
    "Vergütungsstahl": {"kc1.1": 2100, "mc": 0.25, "rate": 0.25, "t_crit": 550},
    "Edelstahl": {"kc1.1": 2400, "mc": 0.22, "rate": 0.45, "t_crit": 650},
    "Titan Grade 5": {"kc1.1": 2800, "mc": 0.24, "rate": 1.2, "t_crit": 750}
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Prozess-Konfiguration")
    mat_name = st.selectbox("Werkstoff-Auswahl", list(MATERIALIEN.keys()))
    m = MATERIALIEN[mat_name]
    vc = st.slider("vc [m/min]", 20, 600, 180)
    f = st.slider("f [mm/U]", 0.01, 1.2, 0.2)
    d = st.number_input("Durchmesser [mm]", 1.0, 100.0, 12.0)
    kss = st.toggle("KSS aktiv", value=True)
    st.divider()
    zyklus_sprung = st.number_input("Schrittweite", 1, 100, 10)
    sim_takt = st.select_slider("Takt (ms)", options=[500, 200, 100, 50, 0], value=100)

# --- 5. PHYSIK-ENGINE ---
s = st.session_state.twin
if s['active'] and not s['broken']:
    s['zyklus'] += zyklus_sprung
    kc = m['kc1.1'] * (f ** -m['mc'])
    s['drehmoment'] = (kc * f * (d/2) * (d/2)) / 1000 
    v_zuwachs = ((m['rate'] * (vc**1.7) * f) / (12000 if kss else 300)) * zyklus_sprung
    s['verschleiss'] += v_zuwachs
    t_ziel = 22 + (s['verschleiss'] * 1.4) + (vc * 0.22) + (0 if kss else 280)
    s['thermik'] += (t_ziel - s['thermik']) * 0.25
    s['vibration'] = (s['verschleiss'] * 0.08) + (vc * 0.015) + s['seed'].normal(0, 0.2)
    s['risk'], evidenz_list, s['rul'] = calculate_metrics(s['zyklus']/1000, s['drehmoment']/60, s['thermik']/m['t_crit'], s['vibration']/10, 1.0 if not kss else 0.0, s['integritaet'])
    
    losses = [ (s['verschleiss']/100)*0.04, (s['drehmoment']/100)*0.01, (np.exp(max(0, s['thermik']-m['t_crit'])/45)-1)*2, (max(0,s['vibration'])/20)*0.05 ]
    s['integritaet'] -= sum(losses) * zyklus_sprung
    if s['integritaet'] <= 0: s['broken'], s['active'], s['integritaet'] = True, False, 0

    exp_text, action_text = get_didactic_explanation(evidenz_list)
    s['logs'].insert(0, {'zeit': time.strftime("%H:%M:%S"), 'risk': s['risk'], 'integ': s['integritaet'], 'exp': exp_text, 'act': action_text, 'scores': evidenz_list[:3]})
    s['history'].append({'z': s['zyklus'], 'i': s['integritaet'], 'r': s['risk'], 't': s['thermik'], 'v': s['vibration']})

# --- 6. UI ---
t1, t2 = st.tabs(["📊 LIVE-ANALYSE", "🧪 LABOR"])
with t1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Integrität", f"{s['integritaet']:.1f}%")
    m2.metric("Bruchrisiko", f"{s['risk']:.1%}")
    m3.metric("Thermik", f"{s['thermik']:.0f}°C")
    m4.metric("Last", f"{s['drehmoment']:.1f}Nm")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        if s['history']:
            df = pd.DataFrame(s['history'])
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Zustand", "KI-Risiko"))
            fig.add_trace(go.Scatter(x=df['z'], y=df['i'], name="Integrität", line=dict(color='#3fb950')), 1, 1)
            fig.add_trace(go.Scatter(x=df['z'], y=df['r']*100, name="Risiko %", line=dict(color='#e3b341')), 2, 1)
            fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 👨‍🏫 Der KI-Meister erklärt:")
        for l in s['logs'][:5]:
            st.markdown(f"""
            <div class="xai-card">
                <div style="display:flex; justify-content:space-between; font-size:12px;">
                    <b style="color:#e3b341;">PROZESS-CHECK {l['zeit']}</b>
                    <b>Gefahr: {l['risk']:.1%}</b>
                </div>
                <div class="reason-text">"{l['exp']}"</div>
                <div class="action-text">👉 Empfehlung: {l['act']}</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()
if st.button("▶ START / STOPP", use_container_width=True): s['active'] = not s['active']
if st.button("🔄 NEUES WERKZEUG", use_container_width=True):
    st.session_state.twin = {'zyklus': 0, 'verschleiss': 0.0, 'history': [], 'logs': [], 'active': False, 'broken': False, 'thermik': 22.0, 'vibration': 0.1, 'risk': 0.0, 'integritaet': 100.0, 'seed': np.random.RandomState(42), 'rul': 800, 'drehmoment': 0.0}
    st.rerun()

if s['active']:
    time.sleep(sim_takt/1000)
    st.rerun()
