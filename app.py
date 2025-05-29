import streamlit as st
import pandas as pd
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.graph_objects as go

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

# ----- HEADER -----
st.markdown(
    "<h1 style='text-align:center;font-size:3.4rem;font-weight:700;color:#232733;margin-bottom:0.25em;margin-top:0.01em;'>Pipeline Optima™ Network Batch Scheduler</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;font-size:2.05rem;font-weight:700;color:#232733;margin-bottom:0.15em;margin-top:0.02em;'>MINLP Pipeline Network Optimization with Batch Scheduling</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.6em; margin-bottom:1.2em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

# ---------------------- INPUT DATA -----------------------
tab_nodes, tab_edges, tab_pumps, tab_peaks, tab_costs = st.tabs([
    "Stations", "Pipe Segments", "Pumps", "Peaks", "Costs & Limits"
])

#### --- NODES ---
with tab_nodes:
    st.subheader("Stations / Demand Centers")
    st.caption("Add all stations (including demand centers). Name must be unique. Demand in m³/month.")
    if "nodes_df" not in st.session_state:
        st.session_state["nodes_df"] = pd.DataFrame(columns=["Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"])
    nodes_df = st.data_editor(
        st.session_state["nodes_df"], num_rows="dynamic", key="nodes_df_editor"
    )
    st.session_state["nodes_df"] = nodes_df
    node_names = list(nodes_df["Name"].dropna().unique())

#### --- EDGES ---
with tab_edges:
    st.subheader("Pipe Segments")
    st.caption("Define each pipe between stations. **Diameter and thickness in inch (auto-converted to m internally).**")
    if "edges_df" not in st.session_state:
        st.session_state["edges_df"] = pd.DataFrame(columns=["From", "To", "Length (km)", "Diameter (inch)", "Thickness (inch)", "Max DR (%)", "Roughness (m)"])
    edges = []
    for idx in range(max(len(st.session_state["edges_df"]), 1)):
        cols = st.columns([1.7,1.7,1,1.4,1.4,1,1])
        from_node = cols[0].selectbox(f"From Station {idx+1}", node_names, key=f"from_{idx}") if node_names else ""
        to_node   = cols[1].selectbox(f"To Station {idx+1}", node_names, key=f"to_{idx}") if node_names else ""
        length    = cols[2].number_input(f"Len (km) {idx+1}", min_value=0.0, value=100.0, step=1.0, key=f"len_{idx}")
        dia_inch  = cols[3].number_input(f"Dia (inch) {idx+1}", min_value=0.0, value=28.0, step=0.5, key=f"dia_{idx}")
        thick_in  = cols[4].number_input(f"Thick (inch) {idx+1}", min_value=0.0, value=0.276, step=0.01, key=f"thick_{idx}")
        max_dr    = cols[5].number_input(f"Max DR (%) {idx+1}", min_value=0.0, value=40.0, step=1.0, key=f"dr_{idx}")
        roughness = cols[6].number_input(f"Rough (m) {idx+1}", min_value=0.0, value=0.00004, step=0.00001, format="%.5f", key=f"rough_{idx}")
        edges.append({"From": from_node, "To": to_node, "Length (km)": length,
                     "Diameter (inch)": dia_inch, "Thickness (inch)": thick_in,
                     "Max DR (%)": max_dr, "Roughness (m)": roughness})
    st.session_state["edges_df"] = pd.DataFrame(edges)

#### --- PUMPS ---
with tab_pumps:
    st.subheader("Pumping Units")
    st.caption("Each row = pump at station for a direction. Upload curves as CSV (Q, Head) or (Q, Eff).")
    if "pumps_df" not in st.session_state:
        st.session_state["pumps_df"] = pd.DataFrame(columns=[
            "Station", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM", "SFC (Diesel)", "Grid Rate (INR/kWh)", "Head Curve CSV", "Eff Curve CSV"
        ])
    pump_entries = []
    for idx in range(max(len(st.session_state["pumps_df"]), 1)):
        cols = st.columns([1.7,1.7,1.2,1,1,1,1.4,1.4])
        stn  = cols[0].selectbox(f"Station {idx+1}", node_names, key=f"pump_stn_{idx}") if node_names else ""
        branch = cols[1].selectbox(f"Branch To {idx+1}", node_names, key=f"branch_to_{idx}") if node_names else ""
        ptype = cols[2].selectbox(f"Power Type {idx+1}", ["Grid", "Diesel"], key=f"ptype_{idx}")
        n_pumps = cols[3].number_input(f"No. Pumps {idx+1}", min_value=1, value=1, step=1, key=f"npumps_{idx}")
        minrpm = cols[4].number_input(f"Min RPM {idx+1}", min_value=0, value=1000, step=50, key=f"minrpm_{idx}")
        maxrpm = cols[5].number_input(f"Max RPM {idx+1}", min_value=0, value=1500, step=50, key=f"maxrpm_{idx}")
        sfc    = cols[6].number_input(f"SFC (Diesel) {idx+1}", min_value=0.0, value=150.0, step=1.0, key=f"sfc_{idx}")
        grid_rate = cols[7].number_input(f"Grid Rate (INR/kWh) {idx+1}", min_value=0.0, value=9.0, step=0.1, key=f"grid_{idx}")
        col9, col10 = st.columns(2)
        head_csv = col9.file_uploader(f"Head Curve CSV {idx+1}", type="csv", key=f"headcsv_{idx}")
        eff_csv = col10.file_uploader(f"Eff Curve CSV {idx+1}", type="csv", key=f"effcsv_{idx}")
        pump_entries.append({
            "Station": stn, "Branch To": branch, "Power Type": ptype, "No. Pumps": n_pumps,
            "Min RPM": minrpm, "Max RPM": maxrpm, "SFC (Diesel)": sfc, "Grid Rate (INR/kWh)": grid_rate,
            "Head Curve CSV": head_csv, "Eff Curve CSV": eff_csv
        })
    st.session_state["pumps_df"] = pd.DataFrame(pump_entries)

#### --- PEAKS ---
with tab_peaks:
    st.subheader("Elevation Peaks")
    st.caption("Specify all elevation peaks BETWEEN two stations (refer 'From'/'To' station as in Pipe Segments). Peak always belongs to a segment.")
    if "peaks_df" not in st.session_state:
        st.session_state["peaks_df"] = pd.DataFrame(columns=["Segment", "Location (km)", "Elevation (m)"])
    peak_entries = []
    segments = [f"{row['From']}→{row['To']}" for _, row in st.session_state["edges_df"].iterrows()]
    for idx in range(max(len(st.session_state["peaks_df"]), 1)):
        cols = st.columns([1.7,1.3,1.3])
        seg = cols[0].selectbox(f"Segment {idx+1}", segments, key=f"peak_seg_{idx}") if segments else ""
        loc = cols[1].number_input(f"Location (km) {idx+1}", min_value=0.0, value=0.0, step=1.0, key=f"peak_loc_{idx}")
        elev = cols[2].number_input(f"Elevation (m) {idx+1}", min_value=0.0, value=0.0, step=1.0, key=f"peak_elev_{idx}")
        peak_entries.append({"Segment": seg, "Location (km)": loc, "Elevation (m)": elev})
    st.session_state["peaks_df"] = pd.DataFrame(peak_entries)

#### --- COSTS, LIMITS, SCHEDULING ---
with tab_costs:
    st.subheader("Global Parameters")
    dra_cost = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
    diesel_price = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    grid_price = st.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
    min_v = st.number_input("Min velocity (m/s)", value=0.5)
    max_v = st.number_input("Max velocity (m/s)", value=3.0)
    time_horizon = st.number_input("Scheduling Horizon (hours)", value=720, step=24)

# ---------------------- VISUALIZATION ----------------------
st.markdown("---")
st.header("Network Preview (Interactive)")

# Generate agraph nodes/edges
agraph_nodes, agraph_edges = [], []
for name in node_names:
    agraph_nodes.append(Node(id=name, label=name, size=30, color="#1f77b4"))
for _, row in st.session_state["edges_df"].iterrows():
    if row["From"] and row["To"]:
        agraph_edges.append(Edge(source=row["From"], target=row["To"], label=f'{row["Length (km)"]} km'))

config = Config(
    width=800,
    height=400,
    directed=True,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#F7A7A6",
    collapsible=False,
)

if agraph_nodes and agraph_edges:
    st.info("Drag the nodes to rearrange. Zoom/pan is supported.")
    agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
else:
    st.warning("Add at least 2 stations and one segment to preview network.")

### ---- BACKEND CALL AND RESULT TABS ----
run = st.button("🚀 Run Network Optimization")
if run:
    with st.spinner("Running optimization (this may take a few minutes on NEOS)..."):
        # ---- PREPARE DATA FOR BACKEND ----
        # (insert your conversion/processing logic here)
        # EXAMPLE: Convert diameter/thickness from inch to m for backend
        edges_backend = []
        for idx, row in edges_df.iterrows():
            edge = dict(row)
            if not pd.isna(edge.get("Diameter (in)")):
                edge["diameter_m"] = float(edge["Diameter (in)"]) * 0.0254
            if not pd.isna(edge.get("Thickness (in)")):
                edge["thickness_m"] = float(edge["Thickness (in)"]) * 0.0254
            edges_backend.append(edge)
        # Same for pumps, stations, peaks as needed.
        # TODO: Build proper dicts for backend

        # MOCK OUTPUT for demo/testing
        results = {
            "flow": {("E1", 1): 120, ("E1", 2): 130},
            "dra": {("E1", 1): 30, ("E1", 2): 40},
            "residual_head": {("A", 1): 60, ("A", 2): 62},
            "pump_on": {("P1", 1): 1, ("P1", 2): 1},
            "pump_rpm": {("P1", 1): 1200, ("P1", 2): 1300},
            "num_pumps": {("P1", 1): 1, ("P1", 2): 1},
            "total_cost": 44500
        }
        st.session_state["results"] = results

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
