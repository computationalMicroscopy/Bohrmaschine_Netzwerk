import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork  # Korrigierter Import
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- BACKEND: KI-MODELL (Bayessches Netzwerk) ---
def setup_model():
    # Struktur definieren mit der neuen Klasse DiscreteBayesianNetwork
    model = DiscreteBayesianNetwork([
        ('Age', 'State'),
        ('Load', 'State'),
        ('Therm', 'State'),
        ('Cool', 'State')
    ])

    # CPDs definieren (0 = Normal/Neu/An, 1 = Alt/Hoch/Aus)
    cpd_age = TabularCPD(variable='Age', variable_card=2, values=[[0.7], [0.3]])
    cpd_load = TabularCPD(variable='Load', variable_card=2, values=[[0.8], [0.2]])
    cpd_therm = TabularCPD(variable='Therm', variable_card=2, values=[[0.9], [0.1]])
    cpd_cool = TabularCPD(variable='Cool', variable_card=2, values=[[0.95], [0.05]])

    # Dynamische State-CPT (Werte gespreizt für deutliche XAI-Sprünge)
    cpd_state = TabularCPD(
        variable='State', variable_card=2,
        values=[
            # P(Bruch=0 | Age, Load, Therm, Cool)
            [0.992, 0.776, 0.718, 0.523, 0.375, 0.153, 0.051, 0.001,
             0.879, 0.652, 0.591, 0.384, 0.212, 0.084, 0.022, 0.000],
            # P(Bruch=1 | Age, Load, Therm, Cool) -> Das Risiko
            [0.008, 0.224, 0.282, 0.477, 0.625, 0.847, 0.949, 0.999,
             0.121, 0.348, 0.409, 0.616, 0.788, 0.916, 0.978, 1.000]
        ],
        evidence=['Age', 'Load', 'Therm', 'Cool'],
        evidence_card=[2, 2, 2, 2]
    )

    model.add_cpds(cpd_age, cpd_load, cpd_therm, cpd_cool, cpd_state)
    return VariableElimination(model)

# --- UI: STREAMLIT INTERFACE ---
st.set_page_config(page_title="KI-Zustandsüberwachung", layout="wide")
st.title("Digitaler Zwilling: Echtzeit-Bruchrisiko-Analyse")

# Sidebar für Parameter
st.sidebar.header("Prozess-Parameter")
material = st.sidebar.selectbox("Werkstoff", ["Baustahl", "Inconel"])
d = st.sidebar.slider("Durchmesser d (mm)", 5, 30, 12)
f = st.sidebar.slider("Vorschub f (mm/U)", 0.05, 0.6, 0.18)
vc = st.sidebar.slider("Schnittgeschwindigkeit vc (m/min)", 50, 250, 160)
cool_on = st.sidebar.toggle("Kühlung Aktiv", value=True)
gain = st.sidebar.slider("Last-Gain (Sensitivität)", 1.0, 5.0, 2.5)
cycles = st.sidebar.number_input("Zykluszähler", 0, 1000, 0)

# Physik-Simulation
k_c11 = 3400 if material == "Inconel" else 1900
md = (k_c11 * (f**0.8) * (d**2)) / 4000
temp_base = (vc * f * k_c11) / 100
current_temp = temp_base if cool_on else temp_base * 3

# Diskretisierung (Trigger)
age_state = 1 if cycles > 650 else 0
load_threshold = (d * 2.2) / gain
load_state = 1 if md > load_threshold else 0
therm_state = 1 if current_temp > 500 else 0
cool_state = 0 if cool_on else 1

# KI-Inferenz
infer = setup_model()
result = infer.query(variables=['State'], evidence={
    'Age': age_state, 'Load': load_state, 'Therm': therm_state, 'Cool': cool_state
})
risk = result.values[1] * 100

# Anzeige
col1, col2 = st.columns(2)
with col1:
    st.metric("Berechnetes Drehmoment Md", f"{md:.2f} Nm")
    st.metric("Aktuelle Temperatur T", f"{current_temp:.1f} °C")
with col2:
    st.metric("KI-Bruchrisiko", f"{risk:.1f} %")

st.subheader("XAI-Logbuch (Echtzeit-Analyse)")
st.code(f"""
[STATUS] Age: {'ALT' if age_state else 'NEU'} (Trigger: Zyklen)
[STATUS] Load: {'HOCH' if load_state else 'NORMAL'} (Trigger: Md)
[STATUS] Therm: {'KRITISCH' if therm_state else 'NORMAL'} (Trigger: Temp)
[STATUS] Cool: {'AUS' if cool_state else 'AN'} (Trigger: Schalter)
--------------------------------------------------
[INFERENZ] Berechnetes Risiko: {risk:.1f}%
""")
