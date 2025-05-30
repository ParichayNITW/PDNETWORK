import streamlit as st
import pandas as pd
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config
from network_batch_pipeline_model import solve_batch_pipeline  # <-- plug in your backend here

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")
st.title("Pipeline Optima™ Network Batch Scheduler")
st.write("Design your pipeline network below. All fields are required unless marked optional. You can add unlimited stations, pipes, pumps, and branches.")
st.markdown("---")

# ----- 1. NODES TABLE -----
st.subheader("Stations / Demand Centers")
if "nodes_df" not in st.session_state:
    st.session_state["nodes_df"] = pd.DataFrame(columns=[
        "Station Name", "Elevation (m)", "Density (kg/m³)", "Viscosity (cSt)", "Monthly Demand (m³)", "Pump Available"
    ])
nodes_df = st.data_editor(
    st.session_state["nodes_df"],
    num_rows="dynamic", key="nodes_df_editor"
)
st.session_state["nodes_df"] = nodes_df
node_names = list(nodes_df["Station Name"].dropna().unique())

st.markdown("---")

# ----- 2. PIPES TABLE -----
st.subheader("Pipe Segments / Branches")
if "pipes_df" not in st.session_state:
    st.session_state["pipes_df"] = pd.DataFrame(columns=[
        "From Node", "To Node", "Length (km)", "Diameter (in)", "Thickness (in)", "Max DR (%)", "Roughness (m)"
    ])
pipes_entries = []
for idx in range(max(1, len(st.session_state["pipes_df"]))):
    col1, col2 = st.columns(2)
    from_node = col1.selectbox(f"From Node {idx+1}", node_names, key=f"pipe_from_{idx}") if node_names else ""
    to_node = col2.selectbox(f"To Node {idx+1}", node_names, key=f"pipe_to_{idx}") if node_names else ""
    col3, col4, col5 = st.columns(3)
    length = col3.number_input(f"Length (km) {idx+1}", min_value=0.0, value=100.0, step=1.0, key=f"len_{idx}")
    diameter = col4.number_input(f"Diameter (in) {idx+1}", min_value=0.0, value=30.0, step=0.5, key=f"dia_{idx}")
    thickness = col5.number_input(f"Thickness (in) {idx+1}", min_value=0.0, value=0.276, step=0.01, key=f"thick_{idx}")
    col6, col7 = st.columns(2)
    max_dr = col6.number_input(f"Max DR (%) {idx+1}", min_value=0.0, value=40.0, step=1.0, key=f"dr_{idx}")
    roughness = col7.number_input(f"Roughness (m) {idx+1}", min_value=0.0, value=0.00004, format="%.5f", step=0.00001, key=f"rough_{idx}")
    pipes_entries.append({
        "From Node": from_node, "To Node": to_node, "Length (km)": length,
        "Diameter (in)": diameter, "Thickness (in)": thickness, "Max DR (%)": max_dr, "Roughness (m)": roughness
    })
st.session_state["pipes_df"] = pd.DataFrame(pipes_entries)
pipes_df = st.session_state["pipes_df"]

st.markdown("---")

# ----- 3. PUMPS TABLE (PER NODE/BRANCH) -----
st.subheader("Pump Details (For Each Branch Where 'Pump Available')")
if "pumps_df" not in st.session_state:
    st.session_state["pumps_df"] = pd.DataFrame(columns=[
        "Station", "Branch To", "Power Type", "No. Pumps", "Min RPM", "Max RPM",
        "SFC (Diesel)", "Grid Rate (INR/kWh)", "Head Curve CSV", "Efficiency Curve CSV"
    ])
pumps_entries = []
for stn in node_names:
    row = nodes_df[nodes_df["Station Name"] == stn]
    if not row.empty and (row.iloc[0].get("Pump Available", False) in [True, 'Yes', 'yes', 1]):
        st.info(f"Add pump(s) for {stn}:")
        for _, branch_row in pipes_df[pipes_df["From Node"] == stn].iterrows():
            branch_to = branch_row["To Node"]
            col1, col2 = st.columns(2)
            power_type = col1.selectbox(f"Power Type ({stn} ➔ {branch_to})", ["Grid", "Diesel"], key=f"ptype_{stn}_{branch_to}")
            n_pumps = col2.number_input(f"No. Pumps ({stn} ➔ {branch_to})", min_value=1, value=1, step=1, key=f"npumps_{stn}_{branch_to}")
            col3, col4 = st.columns(2)
            min_rpm = col3.number_input(f"Min RPM ({stn} ➔ {branch_to})", min_value=0, value=1000, step=50, key=f"minrpm_{stn}_{branch_to}")
            max_rpm = col4.number_input(f"Max RPM ({stn} ➔ {branch_to})", min_value=0, value=1500, step=50, key=f"maxrpm_{stn}_{branch_to}")
            col5, col6 = st.columns(2)
            sfc = col5.number_input(f"SFC (Diesel) ({stn} ➔ {branch_to})", min_value=0.0, value=150.0, step=1.0, key=f"sfc_{stn}_{branch_to}")
            grid_rate = col6.number_input(f"Grid Rate (INR/kWh) ({stn} ➔ {branch_to})", min_value=0.0, value=9.0, step=0.1, key=f"grid_{stn}_{branch_to}")
            head_csv = st.file_uploader(f"Pump Head Curve CSV ({stn} ➔ {branch_to})", type="csv", key=f"headcsv_{stn}_{branch_to}")
            eff_csv = st.file_uploader(f"Pump Eff Curve CSV ({stn} ➔ {branch_to})", type="csv", key=f"effcsv_{stn}_{branch_to}")
            pumps_entries.append({
                "Station": stn, "Branch To": branch_to, "Power Type": power_type, "No. Pumps": n_pumps,
                "Min RPM": min_rpm, "Max RPM": max_rpm, "SFC (Diesel)": sfc, "Grid Rate (INR/kWh)": grid_rate,
                "Head Curve CSV": head_csv, "Efficiency Curve CSV": eff_csv
            })
st.session_state["pumps_df"] = pd.DataFrame(pumps_entries)
pumps_df = st.session_state["pumps_df"]

st.markdown("---")

# ----- 4. PEAKS TABLE (OPTIONAL) -----
st.subheader("Elevation Peaks (Optional)")
if "peaks_df" not in st.session_state:
    st.session_state["peaks_df"] = pd.DataFrame(columns=["Pipe Segment", "Location (km)", "Elevation (m)"])
segment_labels = [f"{row['From Node']}->{row['To Node']}" for _, row in pipes_df.iterrows()]
peaks_entries = []
for idx in range(max(1, len(st.session_state["peaks_df"]))):
    col1 = st.selectbox(f"Pipe Segment {idx+1}", segment_labels, key=f"peakseg_{idx}") if segment_labels else ""
    col2, col3 = st.columns(2)
    loc_km = col2.number_input(f"Location (km) {idx+1}", min_value=0.0, value=0.0, step=0.1, key=f"peakloc_{idx}")
    elev = col3.number_input(f"Elevation (m) {idx+1}", min_value=0.0, value=0.0, step=0.1, key=f"peakelev_{idx}")
    if col1: peaks_entries.append({"Pipe Segment": col1, "Location (km)": loc_km, "Elevation (m)": elev})
st.session_state["peaks_df"] = pd.DataFrame(peaks_entries)
peaks_df = st.session_state["peaks_df"]

st.markdown("---")

# ----- 5. PARAMETERS -----
colA, colB, colC, colD = st.columns(4)
dra_cost = colA.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
diesel_price = colB.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
grid_price = colC.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
time_horizon = colD.number_input("Scheduling Horizon (hours)", value=720, step=24)
min_v = st.number_input("Min velocity (m/s)", value=0.5)
max_v = st.number_input("Max velocity (m/s)", value=3.0)

# ----- 6. NETWORK VISUALIZATION -----
st.subheader("Pipeline Network Visualization")
graph_nodes = [Node(id=row["Station Name"], label=row["Station Name"], size=25, color="#2196f3") for _, row in nodes_df.iterrows() if pd.notna(row["Station Name"])]
graph_edges = [Edge(source=row["From Node"], target=row["To Node"], label=f"{row['From Node']}→{row['To Node']}", color="#666") for _, row in pipes_df.iterrows() if pd.notna(row["From Node"]) and pd.notna(row["To Node"])]
config = Config(width=900, height=500, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True, node={'labelProperty': 'label'}, link={'labelProperty': 'label', "renderLabel": True})
agraph(graph_nodes, graph_edges, config)

st.markdown("---")

# ========== BACKEND CALL AND OUTPUT ==========

# You may want to define utility functions to convert inches to meters here for pipes.

def get_backend_inputs():
    # Prepare dicts for backend, including conversion from inch to meter for diameter/thickness
    def inch_to_m(val): return val * 0.0254 if pd.notna(val) else None
    nodes = []
    for _, row in nodes_df.iterrows():
        nodes.append({
            "id": row["Station Name"], "name": row["Station Name"], "elevation": row["Elevation (m)"],
            "density": row["Density (kg/m³)"], "viscosity": row["Viscosity (cSt)"], "monthly_demand": row["Monthly Demand (m³)"]
        })
    edges = []
    for _, row in pipes_df.iterrows():
        edges.append({
            "id": f"{row['From Node']}_{row['To Node']}",
            "from_node": row["From Node"], "to_node": row["To Node"],
            "length_km": row["Length (km)"], "diameter_m": inch_to_m(row["Diameter (in)"]),
            "thickness_m": inch_to_m(row["Thickness (in)"]), "max_dr": row["Max DR (%)"], "roughness": row["Roughness (m)"]
        })
    # Pump and peaks can be similar
    pumps = []
    for _, row in pumps_df.iterrows():
        pumps.append({
            "station": row["Station"], "branch_to": row["Branch To"], "power_type": row["Power Type"], "n_max": row["No. Pumps"],
            "min_rpm": row["Min RPM"], "max_rpm": row["Max RPM"], "sfc": row["SFC (Diesel)"], "grid_rate": row["Grid Rate (INR/kWh)"],
            # Pass CSVs as file-like objects or parse here if your backend expects arrays
            "head_csv": row["Head Curve CSV"], "eff_csv": row["Efficiency Curve CSV"]
        })
    # Peaks
    peaks = {}
    for _, row in peaks_df.iterrows():
        seg = row["Pipe Segment"]
        if seg:
            from_n, to_n = seg.split("->")
            eid = f"{from_n}_{to_n}"
            if eid not in peaks:
                peaks[eid] = []
            peaks[eid].append({"location_km": row["Location (km)"], "elevation_m": row["Elevation (m)"]})
    # Demands
    demands = {row["Station Name"]: row["Monthly Demand (m³)"] for _, row in nodes_df.iterrows() if pd.notna(row["Monthly Demand (m³)"]) and row["Monthly Demand (m³)"] > 0}
    return nodes, edges, pumps, peaks, demands

if st.button("🚀 Run Batch Network Optimization"):
    with st.spinner("Running MINLP solver... (This may take several minutes)"):
        nodes, edges, pumps, peaks, demands = get_backend_inputs()
        # This is where you pass ALL tables to your backend!
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
        all_edges = list(results["flow"].keys())
        hours = sorted(set(k[1] for k in results["flow"]))
        for e, t in all_edges:
            if t == hours[0]:  # Only once per edge
                tot_flow = sum(results["flow"][(e, th)] for th in hours)
                avg_flow = np.mean([results["flow"][(e, th)] for th in hours])
                tot_dra = sum(results["dra"][(e, th)] for th in hours)
                summary.append({
                    "Edge": e, "Total Vol (m³)": tot_flow, "Avg Flow (m³/hr)": avg_flow,
                    "Avg DRA (%)": tot_dra / len(hours)
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
        st.subheader("Cost Breakdown by Edge")
        df_cost = []
        for e, t in all_edges:
            if t == hours[0]:
                dra_sum = sum(results["dra"][(e, th)] for th in hours)
                flow_sum = sum(results["flow"][(e, th)] for th in hours)
                df_cost.append({
                    "Edge": e, "Total DRA": dra_sum, "Total Flow": flow_sum,
                    "DRA Cost": dra_sum * dra_cost,
                })
        st.dataframe(pd.DataFrame(df_cost))

    # ---- Tab 3: Performance ----
    with tab3:
        st.subheader("Performance (Heads, RH, etc)")
        df_perf = []
        all_nodes = list(set(k[0] for k in results["residual_head"]))
        for n in all_nodes:
            avg_rh = np.mean([results["residual_head"][(n, t)] for t in hours])
            df_perf.append({
                "Node": n, "Avg RH (m)": avg_rh
            })
        st.dataframe(pd.DataFrame(df_perf))

    # ---- Tab 4: System Curves ----
    with tab4:
        st.subheader("System Curves for Selected Edge")
        e_ids = list(set(k[0] for k in results["flow"]))
        selected_e = st.selectbox("Select Edge", e_ids)
        x = hours
        y = [results["flow"][(selected_e, t)] for t in x]
        y2 = [results["dra"][(selected_e, t)] for t in x]
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Flow'))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='DRA'))
        fig.update_layout(title=f"System Curves for {selected_e}", xaxis_title="Hour", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 5: Pump Scheduling ----
    with tab5:
        st.subheader("Pump Operation (ON/OFF, RPM, Num)")
        p_ids = list(set(k[0] for k in results["pump_on"]))
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
        e_ids = list(set(k[0] for k in results["dra"]))
        for e in e_ids:
            y = [results["dra"][(e, t)] for t in hours]
            st.line_chart(y, use_container_width=True)

    # ---- Tab 7: 3D Analysis ----
    with tab7:
        st.subheader("3D Visualization")
        e_ids = list(set(k[0] for k in results["flow"]))
        selected_e = st.selectbox("Edge for 3D", e_ids, key="3d_edge")
        flow_3d = [results["flow"][(selected_e, t)] for t in hours]
        dra_3d = [results["dra"][(selected_e, t)] for t in hours]
        import plotly.graph_objects as go
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
    "<div style='text-align: left; color: #333; margin-top: 2em; font-size: 1.05em;'>"
    "💡 <b>Instructions:</b> <ul>"
    "<li>First add all stations/nodes. Tick 'Pump Available' only for stations where pumps are installed.</li>"
    "<li>Then add pipes: For each segment/branch, select 'From' and 'To' node from drop-down lists.</li>"
    "<li>For each station with a pump, fill all pump details and upload both pump head and efficiency CSV curves.</li>"
    "<li>Optionally, add peaks for any pipe segment. Peaks are intermediate elevation points between two stations.</li>"
    "</ul>"
    "</div>", unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
