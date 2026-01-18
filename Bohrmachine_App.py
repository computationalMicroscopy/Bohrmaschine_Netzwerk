import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. INDUSTRIAL THEME SETUP ---
st.set_page_config(layout="wide", page_title="AI Precision Drilling Lab v14", page_icon="🔩")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .sensor-card { 
        background-color: #161b22; border: 1px solid #30363d; 
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
    .risk-high { color: #f85149 !important; }
    .log-area { font-family: 'Consolas', monospace; font-size: 0.85rem; height: 450px; overflow-y: auto; background: #010409; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ADVANCED CAUSAL ENGINE ---
@st.cache_resource
def build_ultra_bn(n_v, n_t, n_s):
    # Zustand hängt ab von Verschleiß, Last (v_c/f) und Kühlung
    model = DiscreteBayesianNetwork([
        ('Wear', 'Status'), ('Load', 'Status'), ('Cooling', 'Status'),
        ('Status', 'Vib'), ('Status', 'Temp'), ('Status', 'Power')
    ])
    
    # Priors
    cpd_w = TabularCPD('Wear', 3, [[0.33], [0.33], [0.34]]) # Low, Med, High
    cpd_l = TabularCPD('Load', 2, [[0.7], [0.3]])          # Normal, Overload
    cpd_c = TabularCPD('Cooling', 2, [[0.95], [0.05]])     # OK, Fail
    
    # Status CPT (Logic: P(Status|Wear, Load, Cooling))
    # Status: 0=Safe, 1=Warning, 2=Critical/Broken
    s_matrix = []
    for w in range(3):
        for l in range(2):
            for c in range(2):
                risk_score = (w * 2.5) + (l * 4.0) + (c * 6.0)
                p2 = min(0.98, (risk_score**2.2) / 200.0) # Exponentielles Bruchrisiko
                p1 = min(1.0 - p2, risk_score / 15.0)
                p0 = 1.0 - p1 - p2
                s_matrix.append([p0, p1, p2])
                
    cpd_status = TabularCPD('Status', 3, np.array(s_matrix).T, 
                            evidence=['Wear', 'Load', 'Cooling'], evidence_card=[3, 2, 2])
    
    # Sensors (High sensitivity)
    cpd_vib = TabularCPD('Vib', 2, [[1-n_v, 0.4, 0.05], [n_v, 0.6, 0.95]], evidence=['Status'], evidence_card=[3])
    cpd_temp = TabularCPD('Temp', 2, [[0.98, 0.3, 0.01], [0.02, 0.7, 0.99]], evidence=['Status'], evidence_card=[3])
    cpd_pwr = TabularCPD('Power', 2, [[0.9, 0.2, 0.1], [0.1, 0.8, 0.9]], evidence=['Status'], evidence_card=[3])
    
    model.add_cpds(cpd_w, cpd_l, cpd_c, cpd_status, cpd_vib, cpd_temp, cpd_pwr)
    return model

# --- 3. SESSION STATE ---
if 'state' not in st.session_state:
    st.session_state.state = {
        'cycle': 0, 'wear': 0.0, 'history': [], 'logs': [], 
        'active': False, 'broken': False, 'seed': np.random.RandomState(42)
    }

# --- 4. CONTROL INTERFACE ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/automation.png", width=80)
    st.title("Pro-Terminal")
    
    with st.expander("Machine Configuration", expanded=True):
        tool = st.selectbox("Tool Type", ["HSS Co5", "VHM TiAlN Coating"])
        mat = st.selectbox("Workpiece", ["Alu-Cast", "Stainless 1.4301", "Titanium Gr. 5"])
        vc = st.slider("Cutting Speed vc [m/min]", 10, 300, 120)
        f = st.slider("Feed rate f [mm/rev]", 0.02, 0.6, 0.12)
    
    with st.expander("Environmental Noise"):
        n_vib = st.slider("Vibration Noise", 0.0, 1.0, 0.15)
        n_temp = st.slider("Thermal Noise", 0.0, 1.0, 0.05)
        instability = st.slider("Fixture Instability", 0.0, 1.0, 0.1)
    
    cooling = st.toggle("Coolant Active", value=True)
    speed = st.select_slider("Clock Speed [ms]", [500, 200, 100, 50], 100)

# --- 5. CORE SIMULATION ENGINE ---
bn = build_ultra_bn(n_vib, n_temp, 0.05)
infer = VariableElimination(bn)

if st.session_state.state['active'] and not st.session_state.state['broken']:
    s = st.session_state.state
    s['cycle'] += 1
    
    # Real-time Wear Physics (Taylor Equation inspired)
    mat_severity = {"Alu-Cast": 0.05, "Stainless 1.4301": 0.4, "Titanium Gr. 5": 2.2}[mat]
    tool_robustness = 2.5 if "VHM" in tool else 0.8
    load_factor = (vc * f * 10) / 100
    
    wear_step = (mat_severity * load_factor) / (tool_robustness * (10 if cooling else 0.5))
    # Add random micro-fractures
    if s['seed'].rand() > 0.97: wear_step *= 5.0 
    
    s['wear'] += wear_step
    
    # Stochastic Sensors
    current_load = 1 if (vc * f > 30 or s['seed'].rand() > 0.95) else 0
    raw_vib = 10 + (s['wear'] * 0.5) + (instability * 40) + s['seed'].normal(0, 4)
    raw_temp = 25 + (s['wear'] * 0.8) + (vc * 0.2) + (0 if cooling else 130) + s['seed'].normal(0, 2)
    
    # AI Inference
    evidence = {
        'Wear': 0 if s['wear'] < 35 else (1 if s['wear'] < 85 else 2),
        'Load': current_load,
        'Cooling': 0 if cooling else 1,
        'Vib': 1 if raw_vib > 55 else 0,
        'Temp': 1 if raw_temp > 85 else 0
    }
    
    result = infer.query(['Status'], evidence=evidence).values
    risk = result[2]
    
    # Failure Logic (Dynamic Thresholds)
    if risk > 0.96 or s['wear'] > 130 or (risk > 0.7 and s['seed'].rand() > 0.98):
        s['broken'] = True
        s['active'] = False
        
    # Data Recording
    s['history'].append({
        'c': s['cycle'], 'r': risk, 'w': s['wear'], 'v': raw_vib, 't': raw_temp
    })
    s['logs'].insert(0, f"[{time.strftime('%H:%M:%S')}] CYC {s['cycle']:04d} | RISK: {risk:.2%} | TEMP: {raw_temp:.1f}°C")

# --- 6. DASHBOARD RENDERING ---
st.title("🔩 AI Precision Drilling - Digital Twin")
st.caption("Advanced Probabilistic Predictive Maintenance Simulation")

# Action Buttons
c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1: 
    if st.button("▶️ START/PAUSE", use_container_width=True): st.session_state.state['active'] = not st.session_state.state['active']
with c2: 
    if st.button("🔄 SYSTEM RESET", use_container_width=True):
        st.session_state.state = {'cycle':0,'wear':0.0,'history':[],'logs':[],'active':False,'broken':False,'seed':np.random.RandomState(42)}
        st.rerun()
with c3:
    if st.button("💥 FORCE FAIL", use_container_width=True): st.session_state.state['wear'] = 110

if st.session_state.state['broken']:
    st.error(f"🚨 CRITICAL SYSTEM FAILURE: Tool broken at cycle {st.session_state.state['cycle']}. Wear level: {st.session_state.state['wear']:.1f}%")

st.divider()

# Live Metrics
m1, m2, m3, m4 = st.columns(4)
last = st.session_state.state['history'][-1] if st.session_state.state['history'] else {'r':0,'w':0,'v':0,'t':0}

with m1: st.markdown(f'<div class="sensor-card">CYCLES<br><span class="metric-val">{st.session_state.state["cycle"]}</span></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="sensor-card">TOOL WEAR<br><span class="metric-val">{st.session_state.state["wear"]:.1f}%</span></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="sensor-card">VIBRATION<br><span class="metric-val">{last["v"]:.1f}g</span></div>', unsafe_allow_html=True)
with m4:
    risk_color = "risk-high" if last['r'] > 0.7 else ""
    st.markdown(f'<div class="sensor-card">FAILURE RISK<br><span class="metric-val {risk_color}">{last["r"]:.1%}</span></div>', unsafe_allow_html=True)

# Main Visuals
g1, g2 = st.columns([2, 1])

with g1:
    if st.session_state.state['history']:
        df = pd.DataFrame(st.session_state.state['history'])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['c'], y=df['r']*100, name="AI Risk Index (%)", fill='tozeroy', line=dict(color='#f85149', width=3)))
        fig.add_trace(go.Scatter(x=df['c'], y=df['w'], name="Mechanical Wear (%)", line=dict(color='#e3b341', dash='dot')), secondary_y=True)
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="AI Risk Probability", secondary_y=False)
        fig.update_yaxes(title_text="Physical Wear", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("🛠️ Real-Time Telemetry")
    log_text = "".join([f"<div style='margin-bottom:4px; color:{'#f85149' if '9' in l[:20] else '#8b949e'}'>{l}</div>" for l in st.session_state.state['logs'][:50]])
    st.markdown(f'<div class="log-area">{log_text}</div>', unsafe_allow_html=True)

if st.session_state.state['active']:
    time.sleep(speed/1000)
    st.rerun()
