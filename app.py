import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

st.title("Pipeline Optima™ Network Batch Scheduler")
st.markdown("### MINLP Pipeline Network Optimization with Batch Scheduling")
st.markdown("---")

# --------- UTILITY FUNCTIONS ----------
def inch_to_m(x):
    try:
        return float(x) * 0.0254
    except:
        return 0.0

# --------- 1. STATIONS ----------
if "stations" not in st.session_state:
    st.session_state["stations"] = []

st.subheader("Stations")
with st.expander("Add/Edit Stations (Nodes)", expanded=True):
    nodes_df = pd.DataFrame(st.session_state["stations"])
    nodes_df = st.data_editor(
        nodes_df.reindex(columns=["Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"]),
        num_rows="dynamic",
        key="nodes_df"
    )
    st.session_state["stations"] = nodes_df.dropna(subset=["Name"]).to_dict(orient="records")

station_names = [n["Name"] for n in st.session_state["stations"] if n["Name"]]

# --------- 2. PIPE SEGMENTS ----------
if "segments" not in st.session_state:
    st.session_state["segments"] = []

st.subheader("Pipe Segments")
st.markdown("**Diameter and thickness in inch (auto-converted to meters internally).**")
with st.expander("Add/Edit Pipe Segments", expanded=True):
    seg_rows = st.session_state["segments"]
    n_segments = st.number_input("How many pipe segments?", min_value=1, max_value=25, value=max(1, len(seg_rows)), key="n_segments")
    seg_data = []
    for i in range(n_segments):
        cols = st.columns(6)
        from_stn = cols[0].selectbox(f"From Station {i+1}", station_names, key=f"seg_from_{i}")
        to_stn   = cols[1].selectbox(f"To Station {i+1}", station_names, key=f"seg_to_{i}")
        length   = cols[2].number_input(f"Len (km) {i+1}", min_value=0.0, value=100.0, key=f"seg_len_{i}")
        dia_inch = cols[3].number_input(f"Dia (inch) {i+1}", min_value=0.0, value=28.0, key=f"seg_dia_{i}")
        thick_in = cols[4].number_input(f"Thick (inch) {i+1}", min_value=0.0, value=0.28, key=f"seg_thick_{i}")
        max_dr   = cols[5].number_input(f"Max DR (%) {i+1}", min_value=0.0, value=40.0, key=f"seg_dr_{i}")
        rough    = st.number_input(f"Rough (m) {i+1}", min_value=0.0, value=0.00004, format="%.5f", key=f"seg_rough_{i}")
        seg_data.append({
            "From": from_stn, "To": to_stn, "Length (km)": length,
            "Diameter (inch)": dia_inch, "Thickness (inch)": thick_in,
            "Max DR (%)": max_dr, "Roughness (m)": rough
        })
    st.session_state["segments"] = seg_data

# --------- 3. PUMPS ----------
if "pumps" not in st.session_state:
    st.session_state["pumps"] = []

st.subheader("Pumps")
with st.expander("Add/Edit Pumps", expanded=True):
    n_pumps = st.number_input("How many pumps?", min_value=1, max_value=25, value=max(1, len(st.session_state["pumps"])), key="n_pumps")
    pump_data = []
    for i in range(n_pumps):
        cols = st.columns(5)
        stn_name = cols[0].selectbox(f"At Station {i+1}", station_names, key=f"pump_stn_{i}")
        branch_to = cols[1].selectbox(f"Branch To {i+1}", station_names, key=f"pump_to_{i}")
        ptype = cols[2].selectbox(f"Power Type {i+1}", ["Grid", "Diesel"], key=f"pump_type_{i}")
        n_p = cols[3].number_input(f"No. Pumps {i+1}", min_value=1, value=1, key=f"pump_n_{i}")
        minrpm = cols[4].number_input(f"Min RPM {i+1}", min_value=0, value=1000, key=f"pump_minrpm_{i}")
        maxrpm = st.number_input(f"Max RPM {i+1}", min_value=0, value=1500, key=f"pump_maxrpm_{i}")
        sfc    = st.number_input(f"SFC (Diesel) {i+1}", min_value=0.0, value=150.0, key=f"pump_sfc_{i}")
        gridrate = st.number_input(f"Grid Rate (INR/kWh) {i+1}", min_value=0.0, value=9.0, key=f"pump_grid_{i}")
        pump_data.append({
            "Station": stn_name, "Branch To": branch_to, "Power Type": ptype,
            "No. Pumps": n_p, "Min RPM": minrpm, "Max RPM": maxrpm,
            "SFC (Diesel)": sfc, "Grid Rate (INR/kWh)": gridrate
        })
    st.session_state["pumps"] = pump_data

# --------- 4. NETWORK PREVIEW WITH STREAMLIT-AGRAPH ----------
st.markdown("---")
st.subheader("Network Preview (Interactive)")
st.info("Drag the nodes to rearrange. Zoom/pan is supported.", icon="ℹ️")

nodes = []
edges = []
for stn in station_names:
    nodes.append(Node(id=stn, label=stn, size=25))
for seg in st.session_state["segments"]:
    if seg["From"] and seg["To"]:
        edges.append(Edge(source=seg["From"], target=seg["To"], label=f"{seg['From']}-{seg['To']}", width=2))

config = Config(width=1100, height=400, directed=True, physics=True, nodeHighlightBehavior=True,
                highlightColor="#F7A7A6", collapsible=False)
agraph(nodes=nodes, edges=edges, config=config)

### ---- OUTPUT TABS ----
if "results" in st.session_state:
    results = st.session_state["results"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves",
        "🔄 Pump Scheduling", "📉 DRA Curves", "🧊 3D Analysis"
    ])

    with tab1:
        st.subheader("Key Results Table (Example Data)")
        summary = [
            {"Edge": "E1", "From": "A", "To": "B", "Total Vol (m³)": 250, "Avg Flow (m³/hr)": 125, "Avg DRA (%)": 35}
        ]
        st.dataframe(pd.DataFrame(summary))
        st.info(f"Total Optimized Cost (INR): {results['total_cost']:.2f}")
        st.download_button(
            label="Download Results CSV",
            data=pd.DataFrame(summary).to_csv(index=False),
            file_name="results_summary.csv"
        )

    with tab2:
        st.subheader("Cost Breakdown by Node/Edge (Example)")
        df_cost = [
            {"Edge": "E1", "Total DRA": 70, "Total Flow": 250, "DRA Cost": 35000}
        ]
        st.dataframe(pd.DataFrame(df_cost))

    with tab3:
        st.subheader("Performance (Heads, RH, etc)")
        df_perf = [
            {"Node": "A", "Avg RH (m)": 61}
        ]
        st.dataframe(pd.DataFrame(df_perf))

    with tab4:
        st.subheader("System Curves for Selected Edge")
        e_ids = ["E1"]
        selected_e = st.selectbox("Select Edge", e_ids)
        x = [1, 2]
        y = [results["flow"][(selected_e, t)] for t in x]
        y2 = [results["dra"][(selected_e, t)] for t in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Flow'))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='DRA'))
        fig.update_layout(title=f"System Curves for {selected_e}", xaxis_title="Hour", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Pump Operation (ON/OFF, RPM, Num)")
        p_ids = ["P1"]
        selected_p = st.selectbox("Select Pump", p_ids)
        x = [1, 2]
        y_on = [results["pump_on"][(selected_p, t)] for t in x]
        y_rpm = [results["pump_rpm"][(selected_p, t)] for t in x]
        y_n = [results["num_pumps"][(selected_p, t)] for t in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_on, mode='lines+markers', name='Pump ON'))
        fig.add_trace(go.Scatter(x=x, y=y_rpm, mode='lines+markers', name='Pump RPM'))
        fig.add_trace(go.Scatter(x=x, y=y_n, mode='lines+markers', name='Num Pumps'))
        fig.update_layout(title=f"Pump Schedule: {selected_p}", xaxis_title="Hour")
        st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("DRA Dosage Across Edges")
        for e in ["E1"]:
            y = [results["dra"][(e, t)] for t in [1, 2]]
            st.line_chart(y, use_container_width=True)

    with tab7:
        st.subheader("3D Visualization (Example)")
        selected_e = st.selectbox("Edge for 3D", ["E1"], key="3d_edge")
        flow_3d = [results["flow"][(selected_e, t)] for t in [1, 2]]
        dra_3d = [results["dra"][(selected_e, t)] for t in [1, 2]]
        fig = go.Figure(data=[go.Scatter3d(
            x=[1, 2], y=flow_3d, z=dra_3d, mode='lines+markers',
            marker=dict(size=4), line=dict(width=2)
        )])
        fig.update_layout(scene = dict(
            xaxis_title='Hour',
            yaxis_title='Flow (m³/hr)',
            zaxis_title='DRA (%)',
            bgcolor='#222'
        ), title=f"3D Surface: {selected_e}")
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<div style='text-align: center; color: #005bbb; margin-top: 2em; font-size: 1em; font-weight:700;'>"
    "&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
