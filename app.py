import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.graph_objects as go
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import io

# For demo only. Import your actual solver here:
# from network_batch_pipeline_model import solve_batch_pipeline

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

st.markdown(
    "<h1 style='text-align:center;font-size:3.4rem;font-weight:700;color:#005bbb;margin-bottom:0.25em;margin-top:0.01em;'>Pipeline Optima™ Network Batch Scheduler</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;font-size:2.05rem;font-weight:700;color:#444;margin-bottom:0.15em;margin-top:0.02em;'>MINLP Pipeline Network Optimization with Batch Scheduling</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<hr style='margin-top:0.6em; margin-bottom:1.2em; border: 1.5px solid #005bbb;'>",
    unsafe_allow_html=True
)

### ---- SIDEBAR INSTRUCTIONS ----
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown(
        """
- **Stations (Nodes):** Enter all mainline/branch/terminal stations, marketing demand centers. Tick 'Pump Available' if a pump is installed at that node.
- **Pipes:** Connect station pairs, specify physicals (diameter/thickness in INCHES).
- **Pumps:** For each installed pump, specify station, downstream node, type, speed, curves, and upload head/efficiency CSV.
- **Peaks:** For each edge (pipe), optional elevation peaks.
        """
    )
    st.info("Fill all required tables left-to-right, then scroll down to visualize and optimize.")

### ---- DATA ENTRY ----

# 1. Stations Table
st.markdown("### 1. Stations / Demand Centers")
if "nodes_df" not in st.session_state:
    st.session_state["nodes_df"] = pd.DataFrame(columns=[
        "Station Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)", "Pump Available"
    ])
nodes_df = st.session_state["nodes_df"]
nodes_df = st.data_editor(
    nodes_df,
    num_rows="dynamic",
    use_container_width=True,
    key="nodes_editor",
    column_config={
        "Pump Available": st.column_config.CheckboxColumn("Pump Available", default=False)
    }
)
st.session_state["nodes_df"] = nodes_df
node_names = [str(x) for x in nodes_df["Station Name"].dropna().unique()]

# 2. Pipes Table
st.markdown("### 2. Pipe Segments (inches for diameter/thickness)")
if "edges_df" not in st.session_state:
    st.session_state["edges_df"] = pd.DataFrame(columns=[
        "From Station", "To Station", "Length (km)", "Diameter (in)", "Thickness (in)", "Max DR (%)", "Roughness (m)"
    ])
edges_df = st.session_state["edges_df"]
edges_df = st.data_editor(
    edges_df,
    num_rows="dynamic",
    use_container_width=True,
    key="edges_editor",
    column_config={
        "From Station": st.column_config.SelectboxColumn("From Station", options=node_names),
        "To Station": st.column_config.SelectboxColumn("To Station", options=node_names)
    }
)
st.session_state["edges_df"] = edges_df

# 3. Pumps Table
st.markdown("### 3. Pumps (upload head/efficiency CSV for each row)")
if "pumps_df" not in st.session_state:
    st.session_state["pumps_df"] = pd.DataFrame(columns=[
        "Station", "Pumps To", "Power Type", "No. of Pumps", "Min RPM", "Max RPM",
        "SFC (Diesel)", "Grid Rate (INR/kWh)", "Head Curve CSV", "Efficiency Curve CSV"
    ])
pumps_df = st.session_state["pumps_df"]
pumps_df = st.data_editor(
    pumps_df,
    num_rows="dynamic",
    use_container_width=True,
    key="pumps_editor",
    column_config={
        "Station": st.column_config.SelectboxColumn("Station", options=node_names),
        "Pumps To": st.column_config.SelectboxColumn("Pumps To", options=node_names),
        "Power Type": st.column_config.SelectboxColumn("Power Type", options=["Grid", "Diesel"])
    }
)
st.session_state["pumps_df"] = pumps_df

# 4. Elevation Peaks Table
st.markdown("### 4. Elevation Peaks (per Pipe Segment, optional)")
edge_choices = [
    f"{row['From Station']} → {row['To Station']}" for idx, row in st.session_state["edges_df"].iterrows()
]
if "peaks_df" not in st.session_state:
    st.session_state["peaks_df"] = pd.DataFrame(columns=["Pipe Segment", "Location (km)", "Elevation (m)"])
peaks_df = st.session_state["peaks_df"]
peaks_df = st.data_editor(
    peaks_df,
    num_rows="dynamic",
    use_container_width=True,
    key="peaks_editor",
    column_config={
        "Pipe Segment": st.column_config.SelectboxColumn("Pipe Segment", options=edge_choices)
    }
)
st.session_state["peaks_df"] = peaks_df

### ---- PARAMETERS ----
st.markdown("---")
st.markdown("### Global & Cost Parameters")
colA, colB, colC, colD = st.columns(4)
dra_cost = colA.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
diesel_price = colB.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
grid_price = colC.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
time_horizon = colD.number_input("Scheduling Horizon (hours)", value=720, step=24)
colE, colF = st.columns(2)
min_v = colE.number_input("Minimum Velocity (m/s)", value=0.5)
max_v = colF.number_input("Maximum Velocity (m/s)", value=3.0)

### ---- INTERACTIVE NETWORK VISUALIZATION ----
st.markdown("## Network Visualization (drag, zoom, pan)")
nodes_vis = []
edges_vis = []
for i, row in nodes_df.iterrows():
    if pd.isna(row["Station Name"]): continue
    nodes_vis.append(Node(id=row["Station Name"], label=row["Station Name"], shape="ellipse", size=55))
for i, row in edges_df.iterrows():
    if pd.isna(row["From Station"]) or pd.isna(row["To Station"]): continue
    edges_vis.append(Edge(source=row["From Station"], target=row["To Station"], label="", color="#005bbb"))
config = Config(width=900, height=480, directed=True, physics=True, hierarchical=False, nodeHighlightBehavior=True)
agraph(nodes=nodes_vis, edges=edges_vis, config=config)

st.markdown("---")

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
