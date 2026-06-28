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
        "Mikro-Bröckelung der Schneidkante": 0.
