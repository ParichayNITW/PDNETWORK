import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from math import pi
from io import BytesIO
from fpdf import FPDF
import networkx as nx
from network_batch_pipeline_model import solve_batch_pipeline  # import backend!

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

# --- User Inputs ---
with st.sidebar:
    st.title("Pipeline Network Input")
    st.write("**All flow/demand units: m³/hr or m³ (monthly)**")
    # 1. Nodes Table (stations, demand centers, branches)
    st.markdown("#### Nodes (Stations, Demand Centers)")
    nodes_df = st.data_editor(
        pd.DataFrame(columns=["ID", "Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"]),
        num_rows="dynamic", key="nodes_df"
    )
    st.markdown("#### Edges (Pipes/Branches)")
    edges_df = st.data_editor(
        pd.DataFrame(columns=["ID", "From Node", "To Node", "Length (km)", "Diameter (m)", "Thickness (m)", "Max DR (%)", "Roughness (m)"]),
        num_rows="dynamic", key="edges_df"
    )
    st.markdown("#### Pumps")
    pumps_df = st.data_editor(
        pd.DataFrame(columns=[
            "ID", "Node ID", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM",
            "SFC (Diesel)", "Grid Rate (INR/kWh)"
        ]),
        num_rows="dynamic", key="pumps_df"
    )
    st.markdown("#### Peaks (Optional)")
    peaks_df = st.data_editor(
        pd.DataFrame(columns=["Edge ID", "Location (km)", "Elevation (m)"]),
        num_rows="dynamic", key="peaks_df"
    )
    st.markdown("#### Global & Cost")
    dra_cost = st.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
    diesel_price = st.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    grid_price = st.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
    min_v = st.number_input("Min velocity (m/s)", value=0.5)
    max_v = st.number_input("Max velocity (m/s)", value=3.0)
    time_horizon = st.number_input("Scheduling Horizon (hours)", value=720, step=24)
    if st.button("Download Example Input"):
        example = {
            "nodes": [{"id": "A", "name": "A", "elevation": 10, "density": 850, "viscosity": 10, "monthly_demand": 0},
                      {"id": "B", "name": "B", "elevation": 20, "density": 850, "viscosity": 10, "monthly_demand": 15000},
                      {"id": "C", "name": "C", "elevation": 5, "density": 850, "viscosity": 10, "monthly_demand": 12000},
                      {"id": "B1", "name": "B1", "elevation": 18, "density": 850, "viscosity": 10, "monthly_demand": 7000}],
            "edges": [{"id": "E1", "from_node": "A", "to_node": "B", "length_km": 100, "diameter_m": 0.711, "thickness_m": 0.007, "max_dr": 40, "roughness": 0.00004},
                      {"id": "E2", "from_node": "B", "to_node": "C", "length_km": 80, "diameter_m": 0.609, "thickness_m": 0.007, "max_dr": 40, "roughness": 0.00004},
                      {"id": "E3", "from_node": "B", "to_node": "B1", "length_km": 20, "diameter_m": 0.508, "thickness_m": 0.007, "max_dr": 40, "roughness": 0.00004}],
            "pumps": [{"id": "P1", "node_id": "A", "branch_to": "B", "power_type": "Grid", "n_max": 2, "min_rpm": 1100, "max_rpm": 1500, "sfc": 0, "grid_rate": 9.0, "A": -0.0004, "B": 0.35, "C": 50, "P": -2e-8, "Q": 8e-6, "R": -0.001, "S": 0.2, "T": 60},
                      {"id": "P2", "node_id": "B", "branch_to": "C", "power_type": "Diesel", "n_max": 2, "min_rpm": 1000, "max_rpm": 1500, "sfc": 150, "grid_rate": 0, "A": -0.0002, "B": 0.25, "C": 40, "P": -1e-8, "Q": 5e-6, "R": -0.0008, "S": 0.17, "T": 55},
                      {"id": "P3", "node_id": "B", "branch_to": "B1", "power_type": "Grid", "n_max": 1, "min_rpm": 900, "max_rpm": 1350, "sfc": 0, "grid_rate": 9.0, "A": -0.00015, "B": 0.17, "C": 38, "P": -7e-9, "Q": 3e-6, "R": -0.0005, "S": 0.09, "T": 54}]
        }
        st.download_button("Example JSON", data=json.dumps(example, indent=2), file_name="example_network.json", mime="application/json")

# --- Parse input tables ---
def parse_nodes(nodes_df):
    nodes = []
    for _, row in nodes_df.iterrows():
        if row['ID']:
            nodes.append({
                "id": str(row['ID']), "name": str(row['Name']), "elevation": float(row['Elevation (m)']),
                "density": float(row['Density (kg/m³)']), "viscosity": float(row['Viscosity (cSt)'])
            })
    return nodes

def parse_edges(edges_df):
    edges = []
    for _, row in edges_df.iterrows():
        if row['ID'] and row['From Node'] and row['To Node']:
            edges.append({
                "id": str(row['ID']), "from_node": str(row['From Node']), "to_node": str(row['To Node']),
                "length_km": float(row['Length (km)']), "diameter_m": float(row['Diameter (m)']),
                "thickness_m": float(row['Thickness (m)']), "max_dr": float(row['Max DR (%)']),
                "roughness": float(row['Roughness (m)']) if not pd.isna(row['Roughness (m)']) else 0.00004
            })
    return edges

def parse_pumps(pumps_df):
    pumps = []
    for _, row in pumps_df.iterrows():
        if row['ID'] and row['Node ID']:
            pumps.append({
                "id": str(row['ID']), "node_id": str(row['Node ID']),
                "branch_to": str(row['Branch To']) if row['Branch To'] else None,
                "power_type": row['Power Type'], "n_max": int(row['No. Pumps']),
                "min_rpm": int(row['Min RPM']), "max_rpm": int(row['Max RPM']),
                "sfc": float(row['SFC (Diesel)']) if not pd.isna(row['SFC (Diesel)']) else 0.0,
                "grid_rate": float(row['Grid Rate (INR/kWh)']) if not pd.isna(row['Grid Rate (INR/kWh)']) else 0.0,
                # Prompt user to upload or edit pump curves separately if needed
                # These are dummy coefficients for now
                "A": -0.0002, "B": 0.25, "C": 40, "P": -1e-8, "Q": 5e-6, "R": -0.0008, "S": 0.17, "T": 55
            })
    return pumps

def parse_peaks(peaks_df):
    edge_peaks = dict()
    for _, row in peaks_df.iterrows():
        if row['Edge ID']:
            e = str(row['Edge ID'])
            pk = {"location_km": float(row['Location (km)']), "elevation_m": float(row['Elevation (m)'])}
            if e not in edge_peaks:
                edge_peaks[e] = []
            edge_peaks[e].append(pk)
    return edge_peaks

# --- Demand per node ---
def get_demands(nodes_df):
    demands = dict()
    for _, row in nodes_df.iterrows():
        if row['Monthly Demand (m³)'] and float(row['Monthly Demand (m³)']) > 0:
            demands[str(row['ID'])] = float(row['Monthly Demand (m³)'])
    return demands

# --- Visualization: Pipeline Network ---
def visualize_network(nodes, edges):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n['id'], label=n['name'])
    for e in edges:
        G.add_edge(e['from_node'], e['to_node'], label=e['id'])
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    # Edges
    for e in edges:
        x0, y0 = pos[e['from_node']]
        x1, y1 = pos[e['to_node']]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=2, color='black'),
            hoverinfo='none', showlegend=False
        ))
    # Nodes
    for n in nodes:
        x, y = pos[n['id']]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text', marker=dict(size=20, color='#1f77b4'),
            text=n['name'], textposition="bottom center", showlegend=False
        ))
    fig.update_layout(
        plot_bgcolor='#181818', paper_bgcolor='#181818',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title="Pipeline Network Visualization", height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN RUN ---
st.header("Network Preview")
nodes = parse_nodes(nodes_df)
edges = parse_edges(edges_df)
pumps = parse_pumps(pumps_df)
peaks = parse_peaks(peaks_df)
demands = get_demands(nodes_df)
if len(nodes) >= 2 and len(edges) >= 1:
    visualize_network(nodes, edges)

if st.button("🚀 Run Batch Network Optimization"):
    with st.spinner("Running MINLP solver... This may take some time."):
        # Call the backend function
        results = solve_batch_pipeline(
            nodes, edges, pumps, peaks, demands, int(time_horizon),
            dra_cost, diesel_price, grid_price, min_v, max_v
        )
        st.session_state["results"] = results

# ---- OUTPUT TABS (Same 7 Tabs as before!) ----
if "results" in st.session_state:
    results = st.session_state["results"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "💰 Costs", "⚙️ Performance", "🌀 System Curves",
        "🔄 Pump-Scheduling", "📉 DRA Curves", "🧊 3D Analysis"
    ])
    # You can use your straight pipeline code logic for each tab,
    # mapping result keys appropriately to each visualization and output.
    # If you need a working example for each tab, I will post it in the next message!
    with tab1:
        st.subheader("Summary Table")
        # Create and display summary DataFrame, similar to previous logic
        # ...
    # Tabs 2-7: Insert your existing visualization code, mapping to results["flow"], ["dra"], etc.
    # ... (see next message for detailed outputs per tab if needed)

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
