import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SETUP ---
st.set_page_config(layout="wide", page_title="KI-Zwilling Bohrsystem v21.8", page_icon="⚙️")

# --- 2. KI-ENGINE (KORRIGIERTE CPT-LOGIK) ---
@st.cache_resource
def get_engine():
    # Netzwerk-Struktur
    model = DiscreteBayesianNetwork([
        ('Age', 'State'), ('Load', 'State'), 
        ('Therm', 'State'), ('Cool', 'State')
    ])
    
    # Eltern-CPDs (Prior-Wahrscheinlichkeiten)
    cpd_age = TabularCPD('Age', 3, [[0.33], [0.33], [0.34]])
    cpd_load = TabularCPD('Load', 2, [[0.8], [0.2]])
    cpd_therm = TabularCPD('Therm', 2, [[0.9], [0.1]])
    cpd_cool = TabularCPD('Cool', 2, [[0.95], [0.05]])
    
    # KORREKTUR: Systematische Generierung der 24 Kombinationen für 'State'
    # Zustand-Klassen: [Gut, Verschleiß, Kritisch]
    z_matrix = []
    
    for age in range(3):        # 0:Neu, 1:Mittel, 2:Alt
        for load in range(2):    # 0:Normal, 1:Hoch
            for therm in range(2):# 0:Normal, 1:Hoch
                for cool in range(2): # 0:Aktiv, 1:Inaktiv
                    
                    # Basis-Score für Risiko (höher = schlechter)
                    # Gewichtung: Alter (3 Pkt), Last (4 Pkt), Thermik (5 Pkt), Kühlung (6 Pkt)
                    score = (age * 3) + (load * 4) + (therm * 5) + (cool * 6)
                    
                    # Logik zur Verteilung der Wahrscheinlichkeiten
                    if score <= 2:
                        p_gut, p_verschl, p_krit = 0.98, 0.01, 0.01
                    elif score <= 6:
                        p_gut, p_verschl, p_krit = 0.70, 0.25, 0.05
                    elif score <= 10:
                        p_gut, p_verschl, p_krit = 0.30, 0.40, 0.30
                    elif score <= 14:
                        p_gut, p_verschl, p_krit = 0.10, 0.30, 0.60
                    else:
                        p_gut, p_verschl, p_krit = 0.01, 0.04, 0.95
                    
                    z_matrix.append([p_gut, p_verschl, p_krit])
    
    # Hinzufügen der CPT zum Modell
    # Die Matrix muss transponiert werden (.T), um das pgmpy-Format (States x Evidenz-Kombinationen) zu erfüllen
    cpd_state = TabularCPD(
        variable='State', 
        variable_card=3, 
        values=np.array(z_matrix).T,
        evidence=['Age', 'Load', 'Therm', 'Cool'],
        evidence_card=[3, 2, 2, 2]
    )
    
    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state)
    return VariableElimination(model)

# --- 3. INITIALISIERUNG & UI ---
if 'twin' not in st.session_state:
    st.session_state.twin = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 
        'active': False, 'broken': False, 't_current': 22.0, 
        'seed': np.random.RandomState(42)
    }

MATERIALIEN = {
    "Baustahl (S235JR)": {"kc1.1": 1900, "mc": 0.26, "wear_rate": 0.15, "temp_crit": 500},
    "Vergütungsstahl (42CrMo4)": {"kc1.1": 2100, "mc": 0.25, "wear_rate": 0.2, "temp_crit": 550},
    "Edelstahl (rostfrei 1.4404)": {"kc1.1": 2400, "mc": 0.22, "wear_rate": 0.4, "temp_crit": 650},
    "Titan-Legierung (hochfest)": {"kc1.1": 2900, "mc": 0.24, "wear_rate": 1.1, "temp_crit": 750},
    "Inconel (Superlegierung)": {"kc1.1": 3400, "mc": 0.26, "wear_rate": 2.5, "temp_crit": 850}
}

# --- SEITENLEISTE ---
with st.sidebar:
    st.header("⚙️ Parameter")
    mat_name = st.selectbox("Werkstoff", list(MATERIALIEN.keys()))
    mat = MATERIALIEN[mat_name]
    vc = st.slider("Schnittgeschw. vc [m/min]", 20, 500, 160)
    f = st.slider("Vorschub f [mm/U]", 0.02, 1.0, 0.18)
    d = st.number_input("Werkzeug-Ø [mm]", 1.0, 60.0, 12.0)
    cooling = st.toggle("Kühlschmierung", value=True)
    st.divider()
    cycle_step = st.number_input("Schrittweite", min_value=1, max_value=50, value=1)
    sim_speed = st.select_slider("Verzögerung (ms)", options=[500, 200, 100, 50, 10, 0], value=50)

# --- 4. SIMULATIONS-SCHLEIFE ---
if st.session_state.twin['active'] and not st.session_state.twin['broken']:
    s = st.session_state.twin
    s['cycle'] += cycle_step
    
    # Physik
    fc = mat['kc1.1'] * (f** (1-mat['mc'])) * (d/2)
    mc_raw = (fc * d) / 2000 
    s['wear'] += ((mat['wear_rate'] * (vc**1.6) * f) / (15000 if cooling else 400)) * cycle_step
    
    target_t = 22 + (s['wear'] * 1.5) + (vc * 0.2) + (0 if cooling else 250)
    s['t_current'] += (target_t - s['t_current']) * 0.15
    
    # KI-Inferenz
    engine = get_engine()
    
    # Diskretisierung (Grenzwerte)
    age_cat = 0 if s['cycle'] < 250 else (1 if s['cycle'] < 650 else 2)
    load_cat = 1 if mc_raw > (d * 2.2) else 0
    therm_cat = 1 if s['t_current'] > mat['temp_crit'] else 0
    cool_cat = 0 if cooling else 1
    
    risk_query = engine.query(['State'], evidence={
        'Age': age_cat, 'Load': load_cat, 'Therm': therm_cat, 'Cool': cool_cat
    })
    risk = risk_query.values[2] # Wahrscheinlichkeit für 'Kritisch'
    
    if risk > 0.98 or s['wear'] > 100: 
        s['broken'] = True; s['active'] = False
    
    s['history'].append({'c':s['cycle'], 'r':risk, 'w':s['wear'], 't':s['t_current'], 'mc':mc_raw})
    s['logs'].insert(0, f"ZYK {s['cycle']} | Risiko: {risk:.1%} | State-Evidenz: [A:{age_cat}, L:{load_cat}, T:{therm_cat}, C:{cool_cat}]")

# --- 5. UI AUSGABE ---
st.title("KI-ZWILLING | PRÄZISIONS-ÜBERWACHUNG v21.8")
c1, c2, c3 = st.columns(3)
c1.metric("Zyklus", st.session_state.twin['cycle'])
c2.metric("Temperatur", f"{st.session_state.twin['t_current']:.1f} °C")
c3.metric("Verschleiß", f"{st.session_state.twin['wear']:.1f} %")

if len(st.session_state.twin['history']) > 0:
    df = pd.DataFrame(st.session_state.twin['history'])
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="Bruchrisiko %", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['c'], y=df['mc'], name="Drehmoment [Nm]", line=dict(color='blue')), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

st.text_area("Analyse-Protokoll", value="\n".join(st.session_state.twin['logs']), height=200)

if st.button("START / STOPP"): st.session_state.twin['active'] = not st.session_state.twin['active']
if st.button("RESET"): 
    st.session_state.twin = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'t_current':22.0,'seed':np.random.RandomState(42)}
    st.rerun()

if st.session_state.twin['active']:
    time.sleep(sim_speed/1000); st.rerun()
