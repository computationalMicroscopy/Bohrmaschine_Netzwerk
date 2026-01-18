import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time

# --- 1. Definition des Bayesschen Netzwerks ---

def create_bayesian_network():
    # AKTUALISIERT: DiscreteBayesianNetwork statt BayesianNetwork
    model = DiscreteBayesianNetwork([
        ('Alter', 'Zustand'),
        ('Material', 'Zustand'),
        ('Zustand', 'Vibration'),
        ('Zustand', 'Strom'),
        ('Zustand', 'Wartung')
    ])

    # CPTs (Conditional Probability Tables)
    cpd_alter = TabularCPD(variable='Alter', variable_card=3, values=[[0.7], [0.2], [0.1]]) # Neu, Mittel, Alt
    cpd_material = TabularCPD(variable='Material', variable_card=2, values=[[0.5], [0.5]]) # Weich, Hart

    cpd_zustand = TabularCPD(
        variable='Zustand', variable_card=3,
        values=[
            [0.99, 0.90, 0.70, 0.50, 0.05, 0.01], # Intakt
            [0.01, 0.09, 0.25, 0.40, 0.70, 0.30], # Stumpf
            [0.00, 0.01, 0.05, 0.10, 0.25, 0.69]  # Gebrochen
        ],
        evidence=['Alter', 'Material'], evidence_card=[3, 2]
    )

    cpd_vibration = TabularCPD(
        variable='Vibration', variable_card=2,
        values=[[0.90, 0.40, 0.10], [0.10, 0.60, 0.90]],
        evidence=['Zustand'], evidence_card=[3]
    )

    cpd_strom = TabularCPD(
        variable='Strom', variable_card=2,
        values=[[0.95, 0.30, 0.50], [0.05, 0.70, 0.50]],
        evidence=['Zustand'], evidence_card=[3]
    )

    cpd_wartung = TabularCPD(
        variable='Wartung', variable_card=2,
        values=[[0.99, 0.20, 0.05], [0.01, 0.80, 0.95]],
        evidence=['Zustand'], evidence_card=[3]
    )

    model.add_cpds(cpd_alter, cpd_material, cpd_zustand, cpd_vibration, cpd_strom, cpd_wartung)
    model.check_model()
    return model

# --- 2. Simulations-Funktionen ---

def simulate_drilling_step(bn_model, current_alter_state):
    material_state = np.random.choice(2, p=[0.5, 0.5])
    p_intakt = 0.95 - (current_alter_state * 0.15)
    p_stumpf = 0.04 + (current_alter_state * 0.10)
    p_gebrochen = 0.01 + (current_alter_state * 0.05)
    
    total_p = p_intakt + p_stumpf + p_gebrochen
    true_zustand = np.random.choice(3, p=[p_intakt/total_p, p_stumpf/total_p, p_gebrochen/total_p])

    vibration_cpd = bn_model.get_cpds('Vibration')
    simulated_vibration = np.random.choice(2, p=vibration_cpd.values[:, true_zustand])

    strom_cpd = bn_model.get_cpds('Strom')
    simulated_strom = np.random.choice(2, p=strom_cpd.values[:, true_zustand])
    
    return {
        'Alter': current_alter_state, 'Material': material_state,
        'True_Zustand': true_zustand, 'Vibration_Sensor': simulated_vibration,
        'Strom_Sensor': simulated_strom
    }

# --- 3. Streamlit Interface ---

st.set_page_config(layout="wide", page_title="Bohrmaschinen-Diagnose AI")
st.title("⚙️ Standbohrmaschinen-Diagnose (Probabilistisch)")

# Cache das Modell, damit es nicht bei jedem Rerun neu gebaut wird
if 'bn_model' not in st.session_state:
    st.session_state.bn_model = create_bayesian_network()
    st.session_state.inference = VariableElimination(st.session_state.bn_model)

# Session States initialisieren
for key, val in {'drilling_count': 0, 'alter_state': 0, 'history': [], 'is_running': False, 'current_diagnosis': None}.items():
    if key not in st.session_state: st.session_state[key] = val

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Maschinen-Status")
    
    # Visualisierung
    status_label = "Bereit"
    status_color = "gray"
    if st.session_state.current_diagnosis:
        idx = np.argmax(st.session_state.current_diagnosis['Zustand'])
        status_label = ["Intakt", "Stumpf", "Gebrochen"][idx]
        status_color = ["green", "orange", "red"][idx]
    
    st.markdown(f"""
        <div style="border: 3px solid {status_color}; padding: 20px; border-radius: 15px; text-align: center;">
            <h1 style="font-size: 50px; margin: 0;">{"✅" if status_label=="Intakt" else "⚠️" if status_label=="Stumpf" else "❌"}</h1>
            <h2 style="color: {status_color};">{status_label}</h2>
            <p>Bohrvorgänge: {st.session_state.drilling_count} | Alter: {["Neu", "Mittel", "Alt"][st.session_state.alter_state]}</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")
    if st.button("Bohrer wechseln & Reset"):
        st.session_state.drilling_count = 0
        st.session_state.alter_state = 0
        st.session_state.history = []
        st.session_state.current_diagnosis = None
        st.session_state.is_running = False
        st.rerun()

    if st.button("Start Automatik", disabled=st.session_state.is_running):
        st.session_state.is_running = True
        st.rerun()
    if st.button("Stop Automatik", disabled=not st.session_state.is_running):
        st.session_state.is_running = False
        st.rerun()

with col2:
    st.header("Echtzeit-Diagnose (Bayessche Inferenz)")
    
    if st.session_state.is_running:
        # Simulation
        sim = simulate_drilling_step(st.session_state.bn_model, st.session_state.alter_state)
        st.session_state.drilling_count += 1
        
        # Alterung
        if st.session_state.drilling_count > 50: st.session_state.alter_state = 1
        if st.session_state.drilling_count > 150: st.session_state.alter_state = 2

        # Inferenz
        ev = {'Vibration': sim['Vibration_Sensor'], 'Strom': sim['Strom_Sensor'], 
              'Alter': sim['Alter'], 'Material': sim['Material']}
        
        res_z = st.session_state.inference.query(variables=['Zustand'], evidence=ev).values
        res_w = st.session_state.inference.query(variables=['Wartung'], evidence=ev).values
        
        st.session_state.current_diagnosis = {'Zustand': res_z, 'Wartung': res_w}
        
        # History
        st.session_state.history.append({
            'Nr': st.session_state.drilling_count,
            'Prob_Intakt': res_z[0], 'Prob_Stumpf': res_z[1], 'Prob_Gebrochen': res_z[2],
            'Prob_Wartung': res_w[1]
        })

    if st.session_state.current_diagnosis:
        d = st.session_state.current_diagnosis
        st.metric("Wahrscheinlichkeit Defekt (Bruch)", f"{d['Zustand'][2]:.1%}")
        st.progress(float(d['Zustand'][2]))
        
        # Plot
        df_h = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_h['Nr'], y=df_h['Prob_Intakt'], name="Intakt", line_color="green"))
        fig.add_trace(go.Scatter(x=df_h['Nr'], y=df_h['Prob_Stumpf'], name="Stumpf", line_color="orange"))
        fig.add_trace(go.Scatter(x=df_h['Nr'], y=df_h['Prob_Gebrochen'], name="Gebrochen", line_color="red"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.is_running:
        time.sleep(0.3)
        st.rerun()

st.sidebar.markdown("### Info")
st.sidebar.write("Dieses BN modelliert die Kausalität zwischen Bohrer-Alter, Materialwiderstand und Sensorausbrüchen.")
