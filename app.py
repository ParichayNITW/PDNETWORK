import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import json

# Uncomment after backend ready
# from network_batch_pipeline_model import solve_batch_pipeline

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

# ---- HEADER ----
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

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {justify-content: center;}
    .stTabs [data-baseweb="tab"] {font-size:1.2rem;font-weight:700;padding:0.7em 2.5em;}
    .stButton>button {background-color:#005bbb;color:#fff;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- A. NODE ENTRY ----
tab_nodes, tab_edges, tab_pumps, tab_peaks = st.tabs(["🌟 Nodes (Stations/Demand)", "🔗 Segments (Pipes)", "💧 Pumps", "⛰️ Elevation Peaks"])

with tab_nodes:
    st.subheader("Step 1: Define Stations / Demand Centers")
    st.info("Each pipeline node represents a pump station, demand center, or terminal. Provide **Station Name**, **Elevation (m)**, **Fluid Density (kg/m³)**, **Viscosity (cSt)**, and the **monthly demand (m³)** (0 if not a demand node).")
    if "nodes_df" not in st.session_state:
        st.session_state["nodes_df"] = pd.DataFrame(columns=["Station Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)"])
    nodes_df = st.data_editor(
        st.session_state["nodes_df"], num_rows="dynamic", key="nodes_df_editor"
    )
    st.session_state["nodes_df"] = nodes_df
    node_names = list(nodes_df["Station Name"].dropna().unique())

with tab_edges:
    st.subheader("Step 2: Define Pipeline Segments")
    st.info("Each segment connects two stations. **Diameter** and **Thickness** are in **inches** (automatically converted to meters).")
    if "edges_df" not in st.session_state:
        st.session_state["edges_df"] = pd.DataFrame(columns=["From Station", "To Station", "Length (km)", "Diameter (in)", "Thickness (in)", "Max DR (%)", "Roughness (m)"])
    edges_df = st.session_state["edges_df"]
    edge_entries = []
    # For each row, user must pick from dropdowns
    for idx in range(max(len(edges_df), 1)):
        c1, c2 = st.columns(2)
        from_station = c1.selectbox(f"From Station {idx+1}", node_names, key=f"from_{idx}") if node_names else ""
        to_station = c2.selectbox(f"To Station {idx+1}", node_names, key=f"to_{idx}") if node_names else ""
        c3, c4, c5 = st.columns(3)
        length = c3.number_input(f"Length (km) {idx+1}", min_value=0.0, value=100.0, step=1.0, key=f"len_{idx}")
        diameter_in = c4.number_input(f"Diameter (in) {idx+1}", min_value=0.0, value=30.0, step=0.5, key=f"dia_{idx}")
        thickness_in = c5.number_input(f"Thickness (in) {idx+1}", min_value=0.0, value=0.276, step=0.01, key=f"thick_{idx}")
        c6, c7 = st.columns(2)
        max_dr = c6.number_input(f"Max Drag Reducer (%) {idx+1}", min_value=0.0, value=40.0, step=1.0, key=f"dr_{idx}")
        roughness = c7.number_input(f"Roughness (m) {idx+1}", min_value=0.0, value=0.00004, format="%.5f", step=0.00001, key=f"rough_{idx}")
        edge_entries.append({
            "From Station": from_station, "To Station": to_station, "Length (km)": length,
            "Diameter (in)": diameter_in, "Thickness (in)": thickness_in,
            "Max DR (%)": max_dr, "Roughness (m)": roughness
        })
    st.session_state["edges_df"] = pd.DataFrame(edge_entries)
    # For backend, convert inches to meters
    edges = []
    for idx, row in st.session_state["edges_df"].iterrows():
        edges.append({
            "id": f"E{idx+1}",
            "from_node": row["From Station"],
            "to_node": row["To Station"],
            "length_km": row["Length (km)"],
            "diameter_m": row["Diameter (in)"] * 0.0254,
            "thickness_m": row["Thickness (in)"] * 0.0254,
            "max_dr": row["Max DR (%)"],
            "roughness": row["Roughness (m)"]
        })

with tab_pumps:
    st.subheader("Step 3: Define Pumping Units")
    st.info("For each pump, upload the **Head Curve** and **Efficiency Curve** as CSVs (columns: Flow, Head/Efficiency for multiple RPMs).")
    if "pumps_df" not in st.session_state:
        st.session_state["pumps_df"] = pd.DataFrame(columns=[
            "Station", "Pumps to", "Power Type", "No. of Pumps", "Min RPM", "Max RPM",
            "SFC (Diesel)", "Grid Rate (INR/kWh)", "Head Curve CSV", "Efficiency Curve CSV"
        ])
    pumps_entries = []
    for idx in range(max(len(st.session_state["pumps_df"]), 1)):
        c1, c2 = st.columns(2)
        station = c1.selectbox(f"Pump Location {idx+1}", node_names, key=f"pstation_{idx}") if node_names else ""
        branch_to = c2.selectbox(f"Pumps to (Branch) {idx+1}", node_names, key=f"pto_{idx}") if node_names else ""
        c3, c4 = st.columns(2)
        power_type = c3.selectbox(f"Power Type {idx+1}", ["Grid", "Diesel"], key=f"ptype_{idx}")
        n_pumps = c4.number_input(f"No. of Pumps {idx+1}", min_value=1, value=1, step=1, key=f"npumps_{idx}")
        c5, c6 = st.columns(2)
        min_rpm = c5.number_input(f"Min RPM {idx+1}", min_value=0, value=1000, step=50, key=f"minrpm_{idx}")
        max_rpm = c6.number_input(f"Max RPM {idx+1}", min_value=0, value=1500, step=50, key=f"maxrpm_{idx}")
        c7, c8 = st.columns(2)
        sfc = c7.number_input(f"SFC (Diesel) {idx+1}", min_value=0.0, value=150.0, step=1.0, key=f"sfc_{idx}")
        grid_rate = c8.number_input(f"Grid Rate (INR/kWh) {idx+1}", min_value=0.0, value=9.0, step=0.1, key=f"grid_{idx}")
        head_csv = st.file_uploader(f"Head Curve CSV {idx+1} (Flow, Head)", type="csv", key=f"headcsv_{idx}")
        eff_csv = st.file_uploader(f"Efficiency Curve CSV {idx+1} (Flow, Efficiency)", type="csv", key=f"effcsv_{idx}")
        pumps_entries.append({
            "id": f"P{idx+1}", "station": station, "branch_to": branch_to, "power_type": power_type,
            "n_max": n_pumps, "min_rpm": min_rpm, "max_rpm": max_rpm,
            "sfc": sfc, "grid_rate": grid_rate, "head_curve": head_csv, "eff_curve": eff_csv
        })
    st.session_state["pumps_df"] = pd.DataFrame(pumps_entries)

with tab_peaks:
    st.subheader("Step 4: Define Elevation Peaks (Optional)")
    st.info("Each peak is a point of elevation between two connected stations (e.g., 'A → B').")
    edge_choices = [
        f"{row['From Station']} → {row['To Station']}" for idx, row in st.session_state["edges_df"].iterrows()
    ]
    peaks_entries = []
    if "peaks_df" not in st.session_state:
        st.session_state["peaks_df"] = pd.DataFrame(columns=["Pipe Segment", "Location (km)", "Elevation (m)"])
    for idx in range(max(len(st.session_state["peaks_df"]), 1)):
        c1, c2, c3 = st.columns(3)
        segment = c1.selectbox(f"Pipe Segment {idx+1}", edge_choices, key=f"peakedge_{idx}") if edge_choices else ""
        location = c2.number_input(f"Location from Start (km) {idx+1}", min_value=0.0, value=0.0, step=1.0, key=f"peakloc_{idx}")
        elev = c3.number_input(f"Peak Elevation (m) {idx+1}", min_value=0.0, value=0.0, step=0.1, key=f"peakelev_{idx}")
        peaks_entries.append({"Pipe Segment": segment, "Location (km)": location, "Elevation (m)": elev})
    st.session_state["peaks_df"] = pd.DataFrame(peaks_entries)

# ---- OTHER GLOBAL PARAMETERS ----
st.markdown("### Global & Cost Parameters")
colA, colB, colC, colD = st.columns(4)
dra_cost = colA.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
diesel_price = colB.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
grid_price = colC.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
time_horizon = colD.number_input("Scheduling Horizon (hours)", value=720, step=24)
colE, colF = st.columns(2)
min_v = colE.number_input("Minimum Velocity (m/s)", value=0.5)
max_v = colF.number_input("Maximum Velocity (m/s)", value=3.0)

st.markdown("---")

# -------- NETWORK VISUALIZATION ---------
def visualize_network(nodes, edges):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n)
    for e in edges:
        if e["From Station"] and e["To Station"]:
            G.add_edge(e["From Station"], e["To Station"])
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    for e in edges:
        if e["From Station"] and e["To Station"]:
            x0, y0 = pos[e["From Station"]]
            x1, y1 = pos[e["To Station"]]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=2, color='#005bbb'), hoverinfo='none', showlegend=False))
    for n in nodes:
        x, y = pos[n]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', marker=dict(size=24, color='#16a085'),
                                 text=n, textfont=dict(color="#222", size=16), textposition="bottom center", showlegend=False))
    fig.update_layout(
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title="Pipeline Network Visualization", height=500, margin=dict(l=40, r=40, t=70, b=40),
        font=dict(size=16)
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("Network Preview")
nodes = list(nodes_df["Station Name"].dropna().unique())
edges_vis = st.session_state["edges_df"].to_dict(orient="records")
if nodes and len(edges_vis) >= 1:
    visualize_network(nodes, edges_vis)

# --- Data for solver ---
def safe_get(table, idx, key, default=None):
    try:
        val = table.iloc[idx][key]
        if pd.isnull(val):
            return default
        return val
    except Exception:
        return default


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
