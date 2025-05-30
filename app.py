# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from network_batch_pipeline_model import solve_batch_pipeline

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

st.markdown(
    "<h1 style='text-align:center;font-size:3.4rem;font-weight:700;color:#232733;margin-bottom:0.25em;margin-top:0.01em;'>Pipeline Optima™ Network Batch Scheduler</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center;font-size:2.05rem;font-weight:700;color:#232733;margin-bottom:0.15em;margin-top:0.02em;'>MINLP Pipeline Network Optimization with Batch Scheduling</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.6em; margin-bottom:1.2em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

# --- NODES ---
if "nodes_df" not in st.session_state:
    st.session_state["nodes_df"] = pd.DataFrame(columns=["Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"])
nodes_df = st.data_editor(
    st.session_state["nodes_df"], num_rows="dynamic", key="nodes_df_editor"
)
st.session_state["nodes_df"] = nodes_df
node_names = list(nodes_df["Name"].dropna().unique())

# --- EDGES ---
if "edges_df" not in st.session_state:
    st.session_state["edges_df"] = pd.DataFrame(columns=["From Node", "To Node", "Length (km)", "Diameter (in)", "Thickness (in)", "Max DR (%)", "Roughness (m)"])
edges_df = st.session_state["edges_df"]
edge_entries = []
for idx in range(max(len(edges_df), 1)):
    col1, col2 = st.columns(2)
    from_node = col1.selectbox(f"From Node {idx+1}", node_names, key=f"from_node_{idx}") if node_names else ""
    to_node = col2.selectbox(f"To Node {idx+1}", node_names, key=f"to_node_{idx}") if node_names else ""
    col3, col4, col5 = st.columns(3)
    length = col3.number_input(f"Length (km) {idx+1}", min_value=0.0, value=100.0, step=1.0, key=f"len_{idx}")
    diameter = col4.number_input(f"Diameter (in) {idx+1}", min_value=0.0, value=30.0, step=0.01, key=f"dia_{idx}")
    thickness = col5.number_input(f"Thickness (in) {idx+1}", min_value=0.0, value=0.28, step=0.01, key=f"thick_{idx}")
    col6, col7 = st.columns(2)
    max_dr = col6.number_input(f"Max DR (%) {idx+1}", min_value=0.0, value=40.0, step=1.0, key=f"dr_{idx}")
    roughness = col7.number_input(f"Roughness (m) {idx+1}", min_value=0.0, value=0.00004, format="%.5f", step=0.00001, key=f"rough_{idx}")
    edge_entries.append({
        "From Node": from_node, "To Node": to_node, "Length (km)": length, "Diameter_in": diameter,
        "Thickness_in": thickness, "Max DR (%)": max_dr, "Roughness (m)": roughness
    })
st.session_state["edges_df"] = pd.DataFrame(edge_entries)

# --- PUMPS ---
if "pumps_df" not in st.session_state:
    st.session_state["pumps_df"] = pd.DataFrame(columns=[
        "Station", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM", "SFC (Diesel)", "Grid Rate (INR/kWh)", "A", "B", "C", "P", "Q", "R", "S", "T"
    ])
pumps_df = st.session_state["pumps_df"]
pump_entries = []
for idx in range(max(len(pumps_df), 1)):
    col1, col2 = st.columns(2)
    station = col1.selectbox(f"Station {idx+1}", node_names, key=f"pump_stn_{idx}") if node_names else ""
    branch_to = col2.selectbox(f"Branch To {idx+1}", node_names, key=f"branch_to_{idx}") if node_names else ""
    col3, col4 = st.columns(2)
    power_type = col3.selectbox(f"Power Type {idx+1}", ["Grid", "Diesel"], key=f"ptype_{idx}")
    no_pumps = col4.number_input(f"No. Pumps {idx+1}", min_value=1, value=1, step=1, key=f"npumps_{idx}")
    col5, col6 = st.columns(2)
    min_rpm = col5.number_input(f"Min RPM {idx+1}", min_value=0, value=1000, step=50, key=f"minrpm_{idx}")
    max_rpm = col6.number_input(f"Max RPM {idx+1}", min_value=0, value=1500, step=50, key=f"maxrpm_{idx}")
    col7, col8 = st.columns(2)
    sfc = col7.number_input(f"SFC (Diesel) {idx+1}", min_value=0.0, value=150.0, step=1.0, key=f"sfc_{idx}")
    grid_rate = col8.number_input(f"Grid Rate (INR/kWh) {idx+1}", min_value=0.0, value=9.0, step=0.1, key=f"grid_{idx}")
    # Pump curve coefficients (or allow upload, can expand)
    a = st.number_input(f"A (Head curve) {idx+1}", value=-0.0002, format="%.5f", key=f"A_{idx}")
    b = st.number_input(f"B (Head curve) {idx+1}", value=0.25, format="%.5f", key=f"B_{idx}")
    c = st.number_input(f"C (Head curve) {idx+1}", value=40.0, format="%.2f", key=f"C_{idx}")
    p = st.number_input(f"P (Eff curve) {idx+1}", value=-1e-8, format="%.5g", key=f"P_{idx}")
    q = st.number_input(f"Q (Eff curve) {idx+1}", value=5e-6, format="%.5g", key=f"Q_{idx}")
    r = st.number_input(f"R (Eff curve) {idx+1}", value=-0.0008, format="%.5f", key=f"R_{idx}")
    s = st.number_input(f"S (Eff curve) {idx+1}", value=0.17, format="%.2f", key=f"S_{idx}")
    t = st.number_input(f"T (Eff curve) {idx+1}", value=55.0, format="%.2f", key=f"T_{idx}")
    pump_entries.append({
        "station": station, "branch_to": branch_to, "power_type": power_type, "no_pumps": no_pumps,
        "min_rpm": min_rpm, "max_rpm": max_rpm, "sfc": sfc, "grid_rate": grid_rate,
        "A": a, "B": b, "C": c, "P": p, "Q": q, "R": r, "S": s, "T": t
    })
st.session_state["pumps_df"] = pd.DataFrame(pump_entries)

# --- PEAKS ---
if "peaks_df" not in st.session_state:
    st.session_state["peaks_df"] = pd.DataFrame(columns=["Edge", "Location (km)", "Elevation (m)"])
peaks_df = st.data_editor(
    st.session_state["peaks_df"], num_rows="dynamic", key="peaks_df_editor"
)
st.session_state["peaks_df"] = peaks_df

st.markdown("---")
colA, colB, colC, colD = st.columns(4)
dra_cost = colA.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
diesel_price = colB.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
grid_price = colC.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
time_horizon = colD.number_input("Scheduling Horizon (hours)", value=720, step=24)

min_v = st.number_input("Min velocity (m/s)", value=0.5)
max_v = st.number_input("Max velocity (m/s)", value=3.0)

# ----------- NETWORK VISUALIZATION -----------
def visualize_agraph(nodes_df, edges_df):
    nodes = []
    edges = []
    for i, row in nodes_df.iterrows():
        if pd.notna(row["Name"]):
            nodes.append(Node(id=row["Name"], label=row["Name"], size=25, color="#1976d2"))
    for i, row in edges_df.iterrows():
        if pd.notna(row["From Node"]) and pd.notna(row["To Node"]):
            edges.append(Edge(source=row["From Node"], target=row["To Node"], label=f"{row['From Node']}->{row['To Node']}"))
    config = Config(
        width=700,
        height=500,
        directed=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', "renderLabel": True}
    )
    return agraph(nodes=nodes, edges=edges, config=config)

st.header("Network Preview")
if node_names and len(st.session_state["edges_df"]) >= 1:
    visualize_agraph(st.session_state["nodes_df"], st.session_state["edges_df"])

# ----------- PARSE INPUTS FOR BACKEND -----------
def parse_nodes(df):
    return [
        {
            "name": row["Name"], "elevation": float(row["Elevation (m)"]),
            "density": float(row["Density (kg/m³)"]), "viscosity": float(row["Viscosity (cSt)"])
        }
        for _, row in df.iterrows() if pd.notna(row["Name"])
    ]

def parse_edges(df):
    return [
        {
            "from_node": row["From Node"], "to_node": row["To Node"],
            "length_km": float(row["Length (km)"]), "diameter_in": float(row["Diameter_in"]),
            "thickness_in": float(row["Thickness_in"]), "max_dr": float(row["Max DR (%)"]),
            "roughness": float(row["Roughness (m)"])
        }
        for _, row in df.iterrows() if pd.notna(row["From Node"]) and pd.notna(row["To Node"])
    ]

def parse_pumps(df):
    return [
        {
            "station": row["station"], "branch_to": row["branch_to"],
            "power_type": row["power_type"], "no_pumps": int(row["no_pumps"]),
            "min_rpm": int(row["min_rpm"]), "max_rpm": int(row["max_rpm"]),
            "sfc": float(row["sfc"]), "grid_rate": float(row["grid_rate"]),
            "A": float(row["A"]), "B": float(row["B"]), "C": float(row["C"]),
            "P": float(row["P"]), "Q": float(row["Q"]), "R": float(row["R"]), "S": float(row["S"]), "T": float(row["T"])
        }
        for _, row in df.iterrows() if pd.notna(row["station"]) and pd.notna(row["branch_to"])
    ]

def parse_peaks(df):
    peaks = dict()
    for _, row in df.iterrows():
        if pd.notna(row["Edge"]):
            e = row["Edge"]
            if e not in peaks:
                peaks[e] = []
            peaks[e].append({
                "location_km": float(row["Location (km)"]),
                "elevation_m": float(row["Elevation (m)"])
            })
    return peaks

def get_demands(nodes_df):
    demands = {}
    for _, row in nodes_df.iterrows():
        if pd.notna(row["Monthly Demand (m³)"]) and float(row["Monthly Demand (m³)"]) > 0:
            demands[row["Name"]] = float(row["Monthly Demand (m³)"])
    return demands

# ---- RUN OPTIMIZATION ----
if st.button("🚀 Run Batch Network Optimization"):
    nodes = parse_nodes(st.session_state["nodes_df"])
    edges = parse_edges(st.session_state["edges_df"])
    pumps = parse_pumps(st.session_state["pumps_df"])
    peaks = parse_peaks(st.session_state["peaks_df"])
    demands = get_demands(st.session_state["nodes_df"])
    with st.spinner("Running MINLP solver..."):
        results = solve_batch_pipeline(
            nodes, edges, pumps, peaks, demands, int(time_horizon),
            dra_cost, diesel_price, grid_price, min_v, max_v
        )
        st.session_state["results"] = results
# ---- OUTPUT TABS ----
if "results" in st.session_state:
    results = st.session_state["results"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves",
        "🔄 Pump Scheduling", "📉 DRA Curves", "🧊 3D Analysis"
    ])
    # ---- Tab 1: Summary ----
    with tab1:
        st.subheader("Key Results Table")
        summary = []
        all_edges = {f"{e['from_node']}→{e['to_node']}": (e['from_node'], e['to_node']) for e in edges}
        hours = sorted(set(k[1] for k in results["flow"]))
        for e in all_edges:
            tot_flow = sum(results["flow"][(e, t)] for t in hours)
            avg_flow = np.mean([results["flow"][(e, t)] for t in hours])
            tot_dra = sum(results["dra"][(e, t)] for t in hours)
            summary.append({
                "Edge": e, "From": all_edges[e][0], "To": all_edges[e][1],
                "Total Vol (m³)": tot_flow, "Avg Flow (m³/hr)": avg_flow,
                "Avg DRA (%)": tot_dra/len(hours)
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.info(f"Total Optimized Cost (INR): {results['total_cost']:.2f}")
        st.download_button(
            label="Download Results CSV",
            data=pd.DataFrame(summary).to_csv(index=False),
            file_name="results_summary.csv"
        )
    with tab2:
        st.subheader("Cost Breakdown by Node/Edge")
        df_cost = []
        for e in all_edges:
            dra_sum = sum(results["dra"][(e, t)] for t in hours)
            flow_sum = sum(results["flow"][(e, t)] for t in hours)
            df_cost.append({
                "Edge": e, "Total DRA": dra_sum, "Total Flow": flow_sum,
                "DRA Cost": dra_sum * dra_cost,
            })
        st.dataframe(pd.DataFrame(df_cost), use_container_width=True)
    with tab3:
        st.subheader("Performance (Heads, RH, etc)")
        df_perf = []
        for n in node_names:
            avg_rh = np.mean([results["residual_head"][(n, t)] for t in hours])
            df_perf.append({
                "Node": n, "Avg RH (m)": avg_rh
            })
        st.dataframe(pd.DataFrame(df_perf), use_container_width=True)
    with tab4:
        st.subheader("System Curves for Selected Edge")
        e_ids = list(all_edges.keys())
        selected_e = st.selectbox("Select Edge", e_ids)
        x = hours
        y = [results["flow"][(selected_e, t)] for t in x]
        y2 = [results["dra"][(selected_e, t)] for t in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Flow'))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='DRA'))
        fig.update_layout(title=f"System Curves for {selected_e}", xaxis_title="Hour", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
    with tab5:
        st.subheader("Pump Operation (ON/OFF, RPM, Num)")
        p_ids = [p['node_id'] for p in pumps]
        if not p_ids:
            st.warning("No pumps defined.")
        else:
            selected_p = st.selectbox("Select Pump", p_ids)
            x = hours
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
        for e in all_edges:
            y = [results["dra"][(e, t)] for t in hours]
            st.line_chart(y, use_container_width=True)
    with tab7:
        st.subheader("3D Visualization")
        selected_e = st.selectbox("Edge for 3D", list(all_edges.keys()), key="3d_edge")
        flow_3d = [results["flow"][(selected_e, t)] for t in hours]
        dra_3d = [results["dra"][(selected_e, t)] for t in hours]
        fig = go.Figure(data=[go.Scatter3d(
            x=hours, y=flow_3d, z=dra_3d, mode='lines+markers',
            marker=dict(size=3), line=dict(width=2)
        )])
        fig.update_layout(scene = dict(
            xaxis_title='Hour',
            yaxis_title='Flow (m³/hr)',
            zaxis_title='DRA (%)',
            bgcolor='#222'
        ), title=f"3D Surface: {selected_e}")
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
