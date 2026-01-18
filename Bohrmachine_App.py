import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
import time


# --- 1. Definition des Bayesschen Netzwerks (Das Gehirn der Simulation) ---

def create_bayesian_network():
    model = BayesianNetwork([
        ('Alter', 'Zustand'),
        ('Material', 'Zustand'),
        ('Zustand', 'Vibration'),
        ('Zustand', 'Strom'),
        ('Zustand', 'Wartung')  # Neuer Knoten für Wartungsbedarf
    ])

    # Bedingte Wahrscheinlichkeitstabellen (CPTs)
    # ---------------------------------------------
    # P(Alter) - Wir starten meist mit einem neuen Bohrer
    cpd_alter = TabularCPD(
        variable='Alter', variable_card=3, values=[[0.7], [0.2], [0.1]]  # Neu, Mittel, Alt
    )

    # P(Material) - Annahme: gleichverteilt für Simulation
    cpd_material = TabularCPD(
        variable='Material', variable_card=2, values=[[0.5], [0.5]]  # Weich, Hart
    )

    # P(Zustand | Alter, Material)
    # [Intakt, Stumpf, Gebrochen]
    # Alter: Neu, Mittel, Alt
    # Material: Weich, Hart
    cpd_zustand = TabularCPD(
        variable='Zustand', variable_card=3,
        values=[
            # Intakt
            [0.99, 0.90, 0.70,  # Neu & Weich, Neu & Hart, Mittel & Weich
             0.50, 0.05, 0.01],  # Mittel & Hart, Alt & Weich, Alt & Hart
            # Stumpf
            [0.01, 0.09, 0.25,
             0.40, 0.70, 0.30],
            # Gebrochen
            [0.00, 0.01, 0.05,
             0.10, 0.25, 0.69]
        ],
        evidence=['Alter', 'Material'],
        evidence_card=[3, 2]
    )

    # P(Vibration | Zustand)
    # [Niedrig, Hoch]
    # Zustand: Intakt, Stumpf, Gebrochen
    cpd_vibration = TabularCPD(
        variable='Vibration', variable_card=2,
        values=[
            [0.90, 0.40, 0.10],  # Niedrig bei Intakt, Stumpf, Gebrochen
            [0.10, 0.60, 0.90]  # Hoch bei Intakt, Stumpf, Gebrochen
        ],
        evidence=['Zustand'],
        evidence_card=[3]
    )

    # P(Strom | Zustand)
    # [Normal, Hoch]
    # Zustand: Intakt, Stumpf, Gebrochen
    cpd_strom = TabularCPD(
        variable='Strom', variable_card=2,
        values=[
            [0.95, 0.30, 0.50],
            # Normal bei Intakt, Stumpf, Gebrochen (Gebrochen kann auch "Normal" sein, da kein Widerstand)
            [0.05, 0.70, 0.50]  # Hoch bei Intakt, Stumpf, Gebrochen
        ],
        evidence=['Zustand'],
        evidence_card=[3]
    )

    # P(Wartung | Zustand) - Eine hohe Wahrscheinlichkeit für Stumpf oder Gebrochen führt zu Wartung
    # [Nein, Ja]
    # Zustand: Intakt, Stumpf, Gebrochen
    cpd_wartung = TabularCPD(
        variable='Wartung', variable_card=2,
        values=[
            [0.99, 0.20, 0.05],  # Nein bei Intakt, Stumpf, Gebrochen
            [0.01, 0.80, 0.95]  # Ja bei Intakt, Stumpf, Gebrochen
        ],
        evidence=['Zustand'],
        evidence_card=[3]
    )

    model.add_cpds(cpd_alter, cpd_material, cpd_zustand, cpd_vibration, cpd_strom, cpd_wartung)

    # Prüfen, ob das Modell valide ist
    assert model.check_model()
    return model


# --- 2. Simulations-Logik ---

def simulate_drilling_step(current_alter_state):
    """Simuliert einen Bohrvorgang und erzeugt 'Sensor'-Daten."""

    # Simuliere Materialhärte basierend auf P(Material)
    material_state = np.random.choice(2, p=[0.5, 0.5])  # 0: Weich, 1: Hart

    # Simuliere den tatsächlichen Zustand basierend auf Alter und Material
    # Wir nutzen die CPT, um den 'echten' Zustand zu generieren
    alter_idx = current_alter_state
    material_idx = material_state

    # Beispielhaft: Wir schauen in die CPT für Zustand
    # und ziehen eine Stichprobe basierend auf Alter und Material
    cpd_zustand = bn_model.get_cpds('Zustand')
    # Die Index-Berechnung für die CPT ist etwas trickreich:
    # index = material_idx * cpd_zustand.evidence_card[0] + alter_idx

    # Vereinfachte Annahme für Simulation: Alter und Material erhöhen die Wahrscheinlichkeit für schlechten Zustand
    # Für eine exakte Generierung müsste man von der Joint-Verteilung samplen
    p_intakt = 0.95 - (alter_idx * 0.15) - (material_idx * 0.05)
    p_stumpf = 0.04 + (alter_idx * 0.10) + (material_idx * 0.03)
    p_gebrochen = 0.01 + (alter_idx * 0.05) + (material_idx * 0.02)

    # Normalisieren, falls Summe != 1
    total_p = p_intakt + p_stumpf + p_gebrochen
    p_intakt /= total_p
    p_stumpf /= total_p
    p_gebrochen /= total_p

    true_zustand = np.random.choice(3, p=[p_intakt, p_stumpf, p_gebrochen])  # 0: Intakt, 1: Stumpf, 2: Gebrochen

    # Simuliere Vibration und Strom basierend auf dem 'echten' Zustand
    # Hier nutzen wir wieder die CPTs als Sampling-Verteilung
    vibration_cpd = bn_model.get_cpds('Vibration')
    p_vibration = vibration_cpd.values[:, true_zustand]
    simulated_vibration = np.random.choice(2, p=p_vibration)  # 0: Niedrig, 1: Hoch

    strom_cpd = bn_model.get_cpds('Strom')
    p_strom = strom_cpd.values[:, true_zustand]
    simulated_strom = np.random.choice(2, p=p_strom)  # 0: Normal, 1: Hoch

    return {
        'Alter': current_alter_state,
        'Material': material_state,
        'True_Zustand': true_zustand,  # Nur zur Validierung/Analyse
        'Vibration_Sensor': simulated_vibration,
        'Strom_Sensor': simulated_strom
    }


# --- 3. Streamlit App Layout und Logik ---

st.set_page_config(layout="wide", page_title="Bohrmaschinen-Diagnose")

st.title("⚙️ Standbohrmaschinen-Diagnose mit Bayesschem Netzwerk")
st.markdown("Simulieren Sie den Bohrvorgang und beobachten Sie die probabilistische Zustandsdiagnose in Echtzeit.")

# Initialisierung des Bayesschen Netzwerks
bn_model = create_bayesian_network()
inference = VariableElimination(bn_model)

# Session State für die Simulation
if 'drilling_count' not in st.session_state:
    st.session_state.drilling_count = 0
    st.session_state.alter_state = 0  # 0: Neu, 1: Mittel, 2: Alt
    st.session_state.history = []
    st.session_state.current_diagnosis = None
    st.session_state.is_running = False

# Spalten für Layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Bohrmaschinen-Status")

    # Grafische Darstellung der Bohrmaschine (vereinfacht)
    if st.session_state.current_diagnosis:
        # Den wahrscheinlichsten Zustand bestimmen
        zustand_probs = st.session_state.current_diagnosis['Zustand']
        most_likely_zustand_idx = np.argmax(zustand_probs)
        zustand_labels = ["Intakt", "Stumpf", "Gebrochen"]
        aktueller_zustand_str = zustand_labels[most_likely_zustand_idx]

        # Farbe basierend auf dem Zustand
        if aktueller_zustand_str == "Intakt":
            status_color = "green"
        elif aktueller_zustand_str == "Stumpf":
            status_color = "orange"
        else:
            status_color = "red"

        st.markdown(f"""
        <div style="border: 2px solid {status_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h3>Bohrmaschine</h3>
            <p>
                {"✨" if aktueller_zustand_str == "Intakt" else
        "⚠️" if aktueller_zustand_str == "Stumpf" else
        "❌"}
            </p>
            <p style="font-size: 1.2em; font-weight: bold; color: {status_color};">
                Bohrer: {aktueller_zustand_str}
            </p>
            <p>
                Bohrvorgänge: {st.session_state.drilling_count} <br>
                Bohrer-Alter: {["Neu", "Mittel", "Alt"][st.session_state.alter_state]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 2px solid gray; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h3>Bohrmaschine</h3>
            <p>🌀</p>
            <p style="font-size: 1.2em; font-weight: bold; color: gray;">
                Simulation starten...
            </p>
            <p>
                Bohrvorgänge: 0 <br>
                Bohrer-Alter: Neu
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Manuelle Steuerung")
    if st.button("Bohrer tauschen / Reset", help="Setzt den Bohrer auf 'Neu' und den Zähler zurück."):
        st.session_state.drilling_count = 0
        st.session_state.alter_state = 0
        st.session_state.history = []
        st.session_state.current_diagnosis = None
        st.session_state.is_running = False  # Stoppt die automatische Simulation
        st.success("Bohrer getauscht und Simulation zurückgesetzt.")
        st.experimental_rerun()  # Seite neu laden, um Zustand zu aktualisieren

    # Automatischer Simulations-Loop
    st.subheader("Automatische Simulation")
    start_button = st.button("Start Automatische Simulation", disabled=st.session_state.is_running)
    stop_button = st.button("Stopp Automatische Simulation", disabled=not st.session_state.is_running)

    if start_button:
        st.session_state.is_running = True
        st.success("Automatische Simulation gestartet.")
    if stop_button:
        st.session_state.is_running = False
        st.warning("Automatische Simulation gestoppt.")

    st.info("Klicken Sie auf 'Start', um die Bohrvorgänge automatisch ablaufen zu lassen.")

with col2:
    st.header("Probabilistische Diagnose (Bayessches Netzwerk)")
    st.markdown(
        "Die hier gezeigten Wahrscheinlichkeiten sind die **diagnostizierten Zustände** basierend auf den simulierten Sensordaten.")

    # Placeholder für Diagramme und Diagnose
    diagnosis_placeholder = st.empty()
    chart_placeholder = st.empty()
    history_placeholder = st.empty()

    if st.session_state.is_running:
        # Bohrvorgang simulieren
        sim_data = simulate_drilling_step(st.session_state.alter_state)
        st.session_state.drilling_count += 1

        # Alter des Bohrers erhöhen (vereinfacht)
        if st.session_state.drilling_count > 50 and st.session_state.alter_state == 0:
            st.session_state.alter_state = 1  # Mittel
        elif st.session_state.drilling_count > 150 and st.session_state.alter_state == 1:
            st.session_state.alter_state = 2  # Alt

        # Evidenz für das Bayessche Netzwerk
        evidence = {
            'Vibration': sim_data['Vibration_Sensor'],
            'Strom': sim_data['Strom_Sensor'],
            'Alter': sim_data['Alter'],
            'Material': sim_data['Material']
        }

        # Inferenz für den Zustand
        prob_zustand = inference.query(variables=['Zustand'], evidence=evidence).values
        prob_wartung = inference.query(variables=['Wartung'], evidence=evidence).values

        diagnosis_results = {
            'Zustand': prob_zustand,
            'Wartung': prob_wartung
        }
        st.session_state.current_diagnosis = diagnosis_results

        # Speichern in der Historie
        st.session_state.history.append({
            'Count': st.session_state.drilling_count,
            'Alter': ["Neu", "Mittel", "Alt"][sim_data['Alter']],
            'Material': ["Weich", "Hart"][sim_data['Material']],
            'True_Zustand': ["Intakt", "Stumpf", "Gebrochen"][sim_data['True_Zustand']],
            'Sensor_Vibration': ["Niedrig", "Hoch"][sim_data['Vibration_Sensor']],
            'Sensor_Strom': ["Normal", "Hoch"][sim_data['Strom_Sensor']],
            'Prob_Intakt': prob_zustand[0],
            'Prob_Stumpf': prob_zustand[1],
            'Prob_Gebrochen': prob_zustand[2],
            'Prob_Wartung_Ja': prob_wartung[1]
        })

        # Darstellung der aktuellen Diagnose
        with diagnosis_placeholder:
            st.subheader(f"Bohrvorgang #{st.session_state.drilling_count}")
            st.markdown(
                f"**Simulierte Sensordaten:** Vibration: `{['Niedrig', 'Hoch'][sim_data['Vibration_Sensor']]}` | Strom: `{['Normal', 'Hoch'][sim_data['Strom_Sensor']]}` | Material: `{['Weich', 'Hart'][sim_data['Material']]}`")
            st.markdown(
                f"*(Interner 'wahrer' Zustand: `{['Intakt', 'Stumpf', 'Gebrochen'][sim_data['True_Zustand']]}`)*")

            st.write("---")
            st.subheader("Diagnose: Wahrscheinlichkeit für Bohrer-Zustand")
            df_zustand = pd.DataFrame({
                'Zustand': ['Intakt', 'Stumpf', 'Gebrochen'],
                'Wahrscheinlichkeit': prob_zustand
            }).set_index('Zustand')
            st.dataframe(df_zustand.style.format("{:.2%}"), use_container_width=True)

            st.subheader("Wartungsbedarf")
            df_wartung = pd.DataFrame({
                'Bedarf': ['Nein', 'Ja'],
                'Wahrscheinlichkeit': prob_wartung
            }).set_index('Bedarf')
            st.dataframe(df_wartung.style.format("{:.2%}"), use_container_width=True)

            if prob_zustand[2] > 0.5:  # Wenn Wahrscheinlichkeit für Gebrochen über 50%
                st.error("🚨 **WARNUNG: Hohe Wahrscheinlichkeit für Bohrerbruch\!** 🚨")
            elif prob_zustand[1] > 0.7 or prob_wartung[
                1] > 0.8:  # Wenn Wahrscheinlichkeit für Stumpf über 70% oder Wartung über 80%
                st.warning("⚠️ **VORSICHT: Bohrer ist wahrscheinlich stumpf oder benötigt Wartung\!** ⚠️")

        # Dynamische Darstellung des Verlaufs
        df_history = pd.DataFrame(st.session_state.history)
        if not df_history.empty:
            with chart_placeholder:
                st.subheader("Zustands-Wahrscheinlichkeiten über Zeit")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df_history['Count'], y=df_history['Prob_Intakt'], mode='lines', name='Intakt',
                               line=dict(color='green')))
                fig.add_trace(
                    go.Scatter(x=df_history['Count'], y=df_history['Prob_Stumpf'], mode='lines', name='Stumpf',
                               line=dict(color='orange')))
                fig.add_trace(
                    go.Scatter(x=df_history['Count'], y=df_history['Prob_Gebrochen'], mode='lines', name='Gebrochen',
                               line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df_history['Count'], y=df_history['Prob_Wartung_Ja'], mode='lines',
                                         name='Wartung (Ja)', line=dict(color='purple', dash='dot')))
                fig.update_layout(
                    xaxis_title="Bohrvorgänge",
                    yaxis_title="Wahrscheinlichkeit",
                    yaxis_range=[0, 1],
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with history_placeholder:
                st.subheader("Simulationshistorie")
                st.dataframe(df_history.tail(10).style.format({
                    'Prob_Intakt': '{:.2%}',
                    'Prob_Stumpf': '{:.2%}',
                    'Prob_Gebrochen': '{:.2%}',
                    'Prob_Wartung_Ja': '{:.2%}'
                }), use_container_width=True)

        # Kurze Pause für bessere Visualisierung
        time.sleep(0.5)
        st.experimental_rerun()  # Aktualisiert die Seite im Loop

    elif st.session_state.current_diagnosis:  # Wenn die Simulation gestoppt ist, aber Daten vorhanden sind
        # Zeigt die letzte Diagnose an
        diagnosis_results = st.session_state.current_diagnosis
        with diagnosis_placeholder:
            st.subheader(f"Letzter Bohrvorgang #{st.session_state.drilling_count}")
            last_entry = st.session_state.history[-1]
            st.markdown(
                f"**Simulierte Sensordaten:** Vibration: `{last_entry['Sensor_Vibration']}` | Strom: `{last_entry['Sensor_Strom']}` | Material: `{last_entry['Material']}`")
            st.markdown(f"*(Interner 'wahrer' Zustand: `{last_entry['True_Zustand']}`)*")

            st.write("---")
            st.subheader("Diagnose: Wahrscheinlichkeit für Bohrer-Zustand")
            df_zustand = pd.DataFrame({
                'Zustand': ['Intakt', 'Stumpf', 'Gebrochen'],
                'Wahrscheinlichkeit': diagnosis_results['Zustand']
            }).set_index('Zustand')
            st.dataframe(df_zustand.style.format("{:.2%}"), use_container_width=True)

            st.subheader("Wartungsbedarf")
            df_wartung = pd.DataFrame({
                'Bedarf': ['Nein', 'Ja'],
                'Wahrscheinlichkeit': diagnosis_results['Wartung']
            }).set_index('Bedarf')
            st.dataframe(df_wartung.style.format("{:.2%}"), use_container_width=True)

            if diagnosis_results['Zustand'][2] > 0.5:
                st.error("🚨 **WARNUNG: Hohe Wahrscheinlichkeit für Bohrerbruch\!** 🚨")
            elif diagnosis_results['Zustand'][1] > 0.7 or diagnosis_results['Wartung'][1] > 0.8:
                st.warning("⚠️ **VORSICHT: Bohrer ist wahrscheinlich stumpf oder benötigt Wartung\!** ⚠️")

        df_history = pd.DataFrame(st.session_state.history)
        if not df_history.empty:
            with chart_placeholder:
                st.subheader("Zustands-Wahrscheinlichkeiten über Zeit")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df_history['Count'], y=df_history['Prob_Intakt'], mode='lines', name='Intakt',
                               line=dict(color='green')))
                fig.add_trace(
                    go.Scatter(x=df_history['Count'], y=df_history['Prob_Stumpf'], mode='lines', name='Stumpf',
                               line=dict(color='orange')))
                fig.add_trace(
                    go.Scatter(x=df_history['Count'], y=df_history['Prob_Gebrochen'], mode='lines', name='Gebrochen',
                               line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df_history['Count'], y=df_history['Prob_Wartung_Ja'], mode='lines',
                                         name='Wartung (Ja)', line=dict(color='purple', dash='dot')))
                fig.update_layout(
                    xaxis_title="Bohrvorgänge",
                    yaxis_title="Wahrscheinlichkeit",
                    yaxis_range=[0, 1],
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with history_placeholder:
                st.subheader("Simulationshistorie")
                st.dataframe(df_history.tail(10).style.format({
                    'Prob_Intakt': '{:.2%}',
                    'Prob_Stumpf': '{:.2%}',
                    'Prob_Gebrochen': '{:.2%}',
                    'Prob_Wartung_Ja': '{:.2%}'
                }), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Über diese Simulation")
st.sidebar.info(
    "Diese App simuliert eine Standbohrmaschine und verwendet ein "
    "Bayessches Netzwerk, um den Zustand des Bohrers (Intakt, Stumpf, Gebrochen) "
    "und den Wartungsbedarf basierend auf simulierten Sensorwerten "
    "(Vibration, Stromaufnahme) zu diagnostizieren. Die Wahrscheinlichkeiten "
    "basieren auf den vordefinierten 'Conditional Probability Tables' (CPTs)."
)
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Drill_Press_BW_2015-11-09_14-38-00.jpg/640px-Drill_Press_BW_2015-11-09_14-38-00.jpg",
    caption="Generische Standbohrmaschine (Quelle: Wikipedia)")