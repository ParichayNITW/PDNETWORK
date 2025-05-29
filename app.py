import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import networkx as nx
from io import BytesIO
# from network_batch_pipeline_model import solve_batch_pipeline  # Uncomment after backend ready

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

# ----- MAIN INPUT SECTION (Tabs) -----
tab_nodes, tab_edges, tab_pumps, tab_peaks = st.tabs(["Nodes", "Edges", "Pumps", "Peaks"])

with tab_nodes:
    st.subheader("Stations / Demand Centers")
    if "nodes_df" not in st.session_state:
        st.session_state["nodes_df"] = pd.DataFrame(columns=["Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"])
    nodes_df = st.data_editor(
        st.session_state["nodes_df"], num_rows="dynamic", key="nodes_df_editor"
    )
    st.session_state["nodes_df"] = nodes_df

node_names = list(nodes_df["Name"].dropna().unique())

with tab_edges:
    st.subheader("Pipes / Branches")
    if "edges_df" not in st.session_state:
        st.session_state["edges_df"] = pd.DataFrame(columns=["From Node", "To Node", "Length (km)", "Diameter (m)", "Thickness (m)", "Max DR (%)", "Roughness (m)"])
    edges_df = st.session_state["edges_df"]

    # For each row, allow selection from dropdowns for From Node and To Node
    edge_entries = []
    for idx in range(max(len(edges_df), 1)):
        col1, col2 = st.columns(2)
        from_node = col1.selectbox(f"From Node {idx+1}", node_names, key=f"from_node_{idx}") if node_names else ""
        to_node = col2.selectbox(f"To Node {idx+1}", node_names, key=f"to_node_{idx}") if node_names else ""
        col3, col4, col5 = st.columns(3)
        length = col3.number_input(f"Length (km) {idx+1}", min_value=0.0, value=100.0, step=1.0, key=f"len_{idx}")
        diameter = col4.number_input(f"Diameter (m) {idx+1}", min_value=0.0, value=0.762, step=0.01, key=f"dia_{idx}")
        thickness = col5.number_input(f"Thickness (m) {idx+1}", min_value=0.0, value=0.007, step=0.001, key=f"thick_{idx}")
        col6, col7 = st.columns(2)
        max_dr = col6.number_input(f"Max DR (%) {idx+1}", min_value=0.0, value=40.0, step=1.0, key=f"dr_{idx}")
        roughness = col7.number_input(f"Roughness (m) {idx+1}", min_value=0.0, value=0.00004, format="%.5f", step=0.00001, key=f"rough_{idx}")
        edge_entries.append({
            "From Node": from_node, "To Node": to_node, "Length (km)": length, "Diameter (m)": diameter,
            "Thickness (m)": thickness, "Max DR (%)": max_dr, "Roughness (m)": roughness
        })
    st.session_state["edges_df"] = pd.DataFrame(edge_entries)

with tab_pumps:
    st.subheader("Pumping Units")
    if "pumps_df" not in st.session_state:
        st.session_state["pumps_df"] = pd.DataFrame(columns=[
            "Station Name", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM", "SFC (Diesel)", "Grid Rate (INR/kWh)", "Head Curve CSV", "Efficiency Curve CSV"
        ])
    pumps_df = st.session_state["pumps_df"]
    pump_entries = []
    for idx in range(max(len(pumps_df), 1)):
        col1, col2 = st.columns(2)
        stn_name = col1.selectbox(f"Station Name {idx+1}", node_names, key=f"pump_stn_{idx}") if node_names else ""
        branch_to = col2.selectbox(f"Branch To {idx+1}", node_names, key=f"branch_to_{idx}") if node_names else ""
        col3, col4 = st.columns(2)
        power_type = col3.selectbox(f"Power Type {idx+1}", ["Grid", "Diesel"], key=f"ptype_{idx}")
        n_pumps = col4.number_input(f"No. Pumps {idx+1}", min_value=1, value=1, step=1, key=f"npumps_{idx}")
        col5, col6 = st.columns(2)
        min_rpm = col5.number_input(f"Min RPM {idx+1}", min_value=0, value=1000, step=50, key=f"minrpm_{idx}")
        max_rpm = col6.number_input(f"Max RPM {idx+1}", min_value=0, value=1500, step=50, key=f"maxrpm_{idx}")
        col7, col8 = st.columns(2)
        sfc = col7.number_input(f"SFC (Diesel) {idx+1}", min_value=0.0, value=150.0, step=1.0, key=f"sfc_{idx}")
        grid_rate = col8.number_input(f"Grid Rate (INR/kWh) {idx+1}", min_value=0.0, value=9.0, step=0.1, key=f"grid_{idx}")
        head_csv = st.file_uploader(f"Pump Head Curve CSV {idx+1}", type="csv", key=f"headcsv_{idx}")
        eff_csv = st.file_uploader(f"Pump Eff Curve CSV {idx+1}", type="csv", key=f"effcsv_{idx}")
        pump_entries.append({
            "Station Name": stn_name, "Branch To": branch_to, "Power Type": power_type, "No. Pumps": n_pumps,
            "Min RPM": min_rpm, "Max RPM": max_rpm, "SFC (Diesel)": sfc, "Grid Rate (INR/kWh)": grid_rate,
            "Head Curve CSV": head_csv, "Efficiency Curve CSV": eff_csv
        })
    st.session_state["pumps_df"] = pd.DataFrame(pump_entries)

with tab_peaks:
    st.subheader("Elevation Peaks (Optional)")
    edge_ids = [f"E{i+1}" for i in range(len(st.session_state.get("edges_df", [])))]
    if "peaks_df" not in st.session_state:
        st.session_state["peaks_df"] = pd.DataFrame(columns=["Edge ID", "Location (km)", "Elevation (m)"])
    peaks_df = st.data_editor(
        st.session_state["peaks_df"], num_rows="dynamic", key="peaks_df_editor"
    )
    st.session_state["peaks_df"] = peaks_df

st.markdown("---")

# ----- OTHER PARAMETERS -----
colA, colB, colC, colD = st.columns(4)
dra_cost = colA.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
diesel_price = colB.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
grid_price = colC.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
time_horizon = colD.number_input("Scheduling Horizon (hours)", value=720, step=24)

min_v = st.number_input("Min velocity (m/s)", value=0.5)
max_v = st.number_input("Max velocity (m/s)", value=3.0)

# --------- NETWORK VISUALIZATION ---------
def visualize_network(nodes, edges):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for e in edges:
        if e["From Node"] and e["To Node"]:
            G.add_edge(e["From Node"], e["To Node"])
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    for e in edges:
        if e["From Node"] and e["To Node"]:
            x0, y0 = pos[e["From Node"]]
            x1, y1 = pos[e["To Node"]]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=2, color='black'), hoverinfo='none', showlegend=False))
    for n in nodes:
        x, y = pos[n]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', marker=dict(size=20, color='#1f77b4'), text=n, textposition="bottom center", showlegend=False))
    fig.update_layout(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title="Pipeline Network Visualization", height=450, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("Network Preview")
if node_names and len(st.session_state["edges_df"]) >= 1:
    visualize_network(node_names, st.session_state["edges_df"].to_dict(orient="records"))

# ---- Backend Call and Results (Insert backend and output tabs here as before) ----
# Example:
# if st.button("🚀 Run Batch Network Optimization"):
#     ... call your backend ...
#     ... show results tabs as per earlier version ...

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
# ---- RUN OPTIMIZATION ----
if st.button("🚀 Run Batch Network Optimization"):
    with st.spinner("Running MINLP solver... (Can take several minutes)"):
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
        all_nodes = {n['id']: n['name'] for n in nodes}
        all_edges = {e['id']: (e['from_node'], e['to_node']) for e in edges}
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
        st.dataframe(pd.DataFrame(summary))
        st.info(f"Total Optimized Cost (INR): {results['total_cost']:.2f}")
        st.download_button(
            label="Download Results CSV",
            data=pd.DataFrame(summary).to_csv(index=False),
            file_name="results_summary.csv"
        )

    # ---- Tab 2: Cost Breakdown ----
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
        st.dataframe(pd.DataFrame(df_cost))

    # ---- Tab 3: Performance ----
    with tab3:
        st.subheader("Performance (Heads, RH, etc)")
        df_perf = []
        for n in all_nodes:
            avg_rh = np.mean([results["residual_head"][(n, t)] for t in hours])
            df_perf.append({
                "Node": all_nodes[n], "Avg RH (m)": avg_rh
            })
        st.dataframe(pd.DataFrame(df_perf))

    # ---- Tab 4: System Curves ----
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

    # ---- Tab 5: Pump Scheduling ----
    with tab5:
        st.subheader("Pump Operation (ON/OFF, RPM, Num)")
        p_ids = [p['id'] for p in pumps]
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

    # ---- Tab 6: DRA Curves ----
    with tab6:
        st.subheader("DRA Dosage Across Edges")
        for e in all_edges:
            y = [results["dra"][(e, t)] for t in hours]
            st.line_chart(y, use_container_width=True)

    # ---- Tab 7: 3D Analysis ----
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
