import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import io
from network_batch_pipeline_model import solve_batch_pipeline

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

st.markdown(
    "<h1 style='font-size:2.2rem;font-weight:700;color:#232733;'>Pipeline Optima™ Network Batch Scheduler</h1>"
    "<hr style='margin-top:0.2em; margin-bottom:1.2em; border: 1px solid #e1e5ec;'>",
    unsafe_allow_html=True
)

# --- Initialize stateful DataFrames ---
def make_blank_df(cols): return pd.DataFrame(columns=cols)
if "nodes_df" not in st.session_state:
    st.session_state["nodes_df"] = make_blank_df(["Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"])
if "edges_df" not in st.session_state:
    st.session_state["edges_df"] = make_blank_df(["From Node", "To Node", "Length (km)", "Diameter (m)", "Thickness (m)", "Max DR (%)", "Roughness (m)"])
if "pumps_df" not in st.session_state:
    st.session_state["pumps_df"] = make_blank_df(["Station", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM", "SFC (Diesel)", "Grid Rate (INR/kWh)", "Head Curve CSV", "Eff Curve CSV"])
if "peaks_df" not in st.session_state:
    st.session_state["peaks_df"] = make_blank_df(["Edge", "Location (km)", "Elevation (m)"])

input_col, vis_col = st.columns([2.2, 3])

# --- INPUT TABLES (Main area, not sidebar) ---
with input_col:
    st.subheader("Network Input Tables")

    # Nodes
    nodes_df = st.data_editor(st.session_state["nodes_df"], key="nodes_editor", num_rows="dynamic")
    st.session_state["nodes_df"] = nodes_df
    node_names = [n for n in nodes_df["Name"] if pd.notna(n)]

    # Edges: dropdown for node selection
    st.write("#### Edges")
    edges_df = st.session_state["edges_df"]
    for idx in range(len(edges_df)):
        c1, c2 = st.columns(2)
        edges_df.at[idx, "From Node"] = c1.selectbox(f"From Node {idx+1}", node_names,
            index=node_names.index(edges_df.at[idx, "From Node"]) if edges_df.at[idx, "From Node"] in node_names else 0, key=f"edge_from_{idx}")
        edges_df.at[idx, "To Node"] = c2.selectbox(f"To Node {idx+1}", node_names,
            index=node_names.index(edges_df.at[idx, "To Node"]) if edges_df.at[idx, "To Node"] in node_names else 1, key=f"edge_to_{idx}")
    st.data_editor(edges_df, key="edges_editor", num_rows="dynamic")
    st.session_state["edges_df"] = edges_df
    edge_labels = [f"{row['From Node']}→{row['To Node']}" for _, row in edges_df.iterrows() if row['From Node'] and row['To Node']]

    # Peaks: dropdown for edge
    st.write("#### Peaks")
    peaks_df = st.session_state["peaks_df"]
    for idx in range(len(peaks_df)):
        if edge_labels:
            peaks_df.at[idx, "Edge"] = st.selectbox(
                f"Edge for Peak {idx+1}", edge_labels,
                index=edge_labels.index(peaks_df.at[idx, "Edge"]) if peaks_df.at[idx, "Edge"] in edge_labels else 0,
                key=f"peak_edge_{idx}"
            )
    st.data_editor(peaks_df, key="peaks_editor", num_rows="dynamic")
    st.session_state["peaks_df"] = peaks_df

    # Pumps: dropdowns, file uploads for curves, power type
    st.write("#### Pumps")
    pumps_df = st.session_state["pumps_df"]
    for idx in range(len(pumps_df)):
        c1, c2 = st.columns(2)
        if node_names:
            pumps_df.at[idx, "Station"] = c1.selectbox(
                f"Station {idx+1}", node_names,
                index=node_names.index(pumps_df.at[idx, "Station"]) if pumps_df.at[idx, "Station"] in node_names else 0,
                key=f"pump_station_{idx}")
            pumps_df.at[idx, "Branch To"] = c2.selectbox(
                f"Branch To {idx+1}", node_names,
                index=node_names.index(pumps_df.at[idx, "Branch To"]) if pumps_df.at[idx, "Branch To"] in node_names else 0,
                key=f"pump_branch_{idx}")
        pumps_df.at[idx, "Power Type"] = st.selectbox(f"Power Type {idx+1}", ["Grid", "Diesel"],
            index=0 if pumps_df.at[idx, "Power Type"]=="Grid" else 1,
            key=f"power_type_{idx}")
        # Head/Eff curve upload
        pumps_df.at[idx, "Head Curve CSV"] = st.file_uploader(f"Head Curve (CSV, Flow,Head) {idx+1}", type="csv", key=f"head_csv_{idx}")
        pumps_df.at[idx, "Eff Curve CSV"] = st.file_uploader(f"Eff Curve (CSV, Flow,Eff) {idx+1}", type="csv", key=f"eff_csv_{idx}")
    st.data_editor(pumps_df, key="pumps_editor", num_rows="dynamic")
    st.session_state["pumps_df"] = pumps_df

    # Global/Cost
    st.write("#### Global Parameters")
    dra_cost = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
    diesel_price = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    grid_price = st.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
    min_v = st.number_input("Min velocity (m/s)", value=0.5)
    max_v = st.number_input("Max velocity (m/s)", value=3.0)
    time_horizon = st.number_input("Scheduling Horizon (hours)", value=720, step=24)

# --- VISUALIZATION ---
with vis_col:
    st.subheader("Network Visualization")
    try:
        nodes = [n for n in node_names if n]
        G = nx.DiGraph()
        for n in nodes: G.add_node(n)
        for _, row in st.session_state["edges_df"].iterrows():
            if row["From Node"] and row["To Node"]:
                G.add_edge(row["From Node"], row["To Node"])
        pos = nx.spring_layout(G, seed=42)
        fig = go.Figure()
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=2, color='black'),
                hoverinfo='none', showlegend=False
            ))
        for node in G.nodes():
            x, y = pos[node]
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text', marker=dict(size=20, color='#1f77b4'),
                text=node, textposition="bottom center", showlegend=False
            ))
        fig.update_layout(
            plot_bgcolor='#fff', paper_bgcolor='#fff',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            title="Pipeline Network", height=450, margin=dict(l=10, r=10, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as ex:
        st.warning("Network not visualized (check input data).")

st.markdown("---")

# --- OPTIMIZATION RUN BUTTON ---
if st.button("🚀 Run Batch Network Optimization", type="primary"):
    nodes = []
    for _, row in st.session_state["nodes_df"].dropna(subset=["Name"]).iterrows():
        nodes.append({
            "id": row["Name"], "name": row["Name"], "elevation": float(row["Elevation (m)"]),
            "density": float(row["Density (kg/m³)"]), "viscosity": float(row["Viscosity (cSt)"]),
            "monthly_demand": float(row["Monthly Demand (m³)"]) if not pd.isna(row["Monthly Demand (m³)"]) else 0
        })
    edges = []
    for _, row in st.session_state["edges_df"].dropna(subset=["From Node", "To Node"]).iterrows():
        edges.append({
            "id": f"{row['From Node']}_{row['To Node']}", "from_node": row["From Node"], "to_node": row["To Node"],
            "length_km": float(row["Length (km)"]), "diameter_m": float(row["Diameter (m)"]),
            "thickness_m": float(row["Thickness (m)"]), "max_dr": float(row["Max DR (%)"]),
            "roughness": float(row["Roughness (m)"])
        })
    peaks = {}
    for _, row in st.session_state["peaks_df"].dropna(subset=["Edge"]).iterrows():
        key = row["Edge"]
        if key not in peaks:
            peaks[key] = []
        peaks[key].append({"location_km": float(row["Location (km)"]), "elevation_m": float(row["Elevation (m)"])})
    pumps = []
    for idx, row in st.session_state["pumps_df"].dropna(subset=["Station", "Branch To"]).iterrows():
        pump = {
            "id": f"P{idx+1}", "node_id": row["Station"], "branch_to": row["Branch To"],
            "power_type": row["Power Type"], "n_max": int(row["No. Pumps"]),
            "min_rpm": int(row["Min RPM"]), "max_rpm": int(row["Max RPM"]),
            "sfc": float(row["SFC (Diesel)"]), "grid_rate": float(row["Grid Rate (INR/kWh)"])
        }
        # TODO: Extract head/eff curve coefficients from uploaded CSVs if provided
        pump["A"], pump["B"], pump["C"] = -0.0002, 0.25, 40
        pump["P"], pump["Q"], pump["R"], pump["S"], pump["T"] = -1e-8, 5e-6, -0.0008, 0.17, 55
        pumps.append(pump)
    demands = {n["id"]: n["monthly_demand"] for n in nodes if n["monthly_demand"] > 0}
    results = solve_batch_pipeline(
        nodes, edges, pumps, peaks, demands, int(time_horizon),
        dra_cost, diesel_price, grid_price, min_v, max_v
    )
    st.session_state["results"] = results

# --- OUTPUT TABS ---
if "results" in st.session_state:
    results = st.session_state["results"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves",
        "🔄 Pump Scheduling", "📉 DRA Curves", "🧊 3D Analysis"
    ])
    with tab1: st.write("Summary Table (implement as needed)")
    with tab2: st.write("Cost Breakdown (implement as needed)")
    with tab3: st.write("Performance (implement as needed)")
    with tab4: st.write("System Curves (implement as needed)")
    with tab5: st.write("Pump Scheduling (implement as needed)")
    with tab6: st.write("DRA Curves (implement as needed)")
    with tab7: st.write("3D Analysis (implement as needed)")

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
