import streamlit as st
import pandas as pd
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.graph_objects as go
import json
from network_batch_pipeline_model import solve_batch_pipeline  # <-- Your backend

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

st.markdown(
    "<h1 style='text-align:center;font-size:2.6rem;font-weight:700;color:#232733;margin-bottom:0.15em;'>Pipeline Optima™ Network Batch Scheduler</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.4em; margin-bottom:1.0em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

# -------------------------------
# ---- DATA ENTRY -------------
# -------------------------------

# 1. NODES
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []
st.subheader("Stations / Demand Centers")
with st.form("add_node_form", clear_on_submit=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    name = c1.text_input("Station Name")
    elevation = c2.number_input("Elevation (m)")
    density = c3.number_input("Density (kg/m³)", value=850.0)
    viscosity = c4.number_input("Viscosity (cSt)", value=10.0)
    monthly_demand = c5.number_input("Monthly Demand (m³)", value=0.0)
    add = st.form_submit_button("Add Station")
    if add and name:
        st.session_state["nodes"].append({
            "Name": name, "Elevation (m)": elevation,
            "Density (kg/m³)": density, "Viscosity (cSt)": viscosity,
            "Monthly Demand (m³)": monthly_demand
        })
if st.session_state["nodes"]:
    st.dataframe(pd.DataFrame(st.session_state["nodes"]), use_container_width=True)
    if st.button("Clear All Stations"):
        st.session_state["nodes"] = []

node_names = [n["Name"] for n in st.session_state["nodes"]]
st.divider()

# 2. PIPE SEGMENTS (EDGES)
if "edges" not in st.session_state:
    st.session_state["edges"] = []
st.subheader("Pipe Segments")
with st.form("add_edge_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    from_node = c1.selectbox("From Node", node_names, key="from_node_edge")
    to_node = c2.selectbox("To Node", node_names, key="to_node_edge")
    length = c3.number_input("Length (km)", min_value=0.0, value=10.0)
    c4, c5, c6 = st.columns(3)
    dia_in = c4.number_input("Diameter (inches)", min_value=0.0, value=28.0)
    thick_in = c5.number_input("Thickness (inches)", min_value=0.0, value=0.276)
    rough = c6.number_input("Roughness (m)", min_value=0.0, value=0.00004, format="%.5f")
    max_dr = st.number_input("Max Drag Reducer (%)", min_value=0.0, value=40.0)
    add_edge = st.form_submit_button("Add Pipe Segment")
    if add_edge and from_node and to_node:
        st.session_state["edges"].append({
            "From Node": from_node, "To Node": to_node, "Length (km)": length,
            "Diameter (inches)": dia_in, "Thickness (inches)": thick_in,
            "Roughness (m)": rough, "Max DR (%)": max_dr
        })
if st.session_state["edges"]:
    st.dataframe(pd.DataFrame(st.session_state["edges"]), use_container_width=True)
    if st.button("Clear All Pipe Segments"):
        st.session_state["edges"] = []
segment_pairs = [f"{e['From Node']} → {e['To Node']}" for e in st.session_state["edges"]]
st.divider()

# 3. PUMPS
if "pumps" not in st.session_state:
    st.session_state["pumps"] = []
st.subheader("Pumping Units")
with st.form("add_pump_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    stn_name = c1.selectbox("Station Name", node_names, key="pump_stn_name2")
    branch_to = c2.selectbox("Branch To", node_names, key="pump_branch_to2")
    power_type = c3.selectbox("Power Type", ["Grid", "Diesel"])
    c4, c5, c6 = st.columns(3)
    n_pumps = c4.number_input("No. Pumps", min_value=1, value=1, step=1)
    min_rpm = c5.number_input("Min RPM", min_value=0, value=1000, step=50)
    max_rpm = c6.number_input("Max RPM", min_value=0, value=1500, step=50)
    sfc = st.number_input("SFC (Diesel)", min_value=0.0, value=150.0, step=1.0)
    grid_rate = st.number_input("Grid Rate (INR/kWh)", min_value=0.0, value=9.0, step=0.1)
    head_csv = st.file_uploader("Pump Head Curve CSV", type="csv")
    eff_csv = st.file_uploader("Pump Efficiency Curve CSV", type="csv")
    add_pump = st.form_submit_button("Add Pump")
    if add_pump and stn_name and branch_to:
        st.session_state["pumps"].append({
            "Station Name": stn_name, "Branch To": branch_to, "Power Type": power_type,
            "No. Pumps": n_pumps, "Min RPM": min_rpm, "Max RPM": max_rpm,
            "SFC (Diesel)": sfc, "Grid Rate (INR/kWh)": grid_rate,
            "Head Curve CSV": head_csv.name if head_csv else "",
            "Efficiency Curve CSV": eff_csv.name if eff_csv else ""
        })
if st.session_state["pumps"]:
    st.dataframe(pd.DataFrame(st.session_state["pumps"]), use_container_width=True)
    if st.button("Clear All Pumps"):
        st.session_state["pumps"] = []
st.divider()

# 4. PEAKS
if "peaks" not in st.session_state:
    st.session_state["peaks"] = []
st.subheader("Elevation Peaks (Optional)")
with st.form("add_peak_form", clear_on_submit=True):
    segment = st.selectbox("Pipe Segment", segment_pairs)
    loc_km = st.number_input("Location (km, from start of segment)", min_value=0.0, value=0.0)
    elev = st.number_input("Elevation at Peak (m)", min_value=0.0, value=0.0)
    add_peak = st.form_submit_button("Add Peak")
    if add_peak and segment:
        st.session_state["peaks"].append({
            "Segment": segment, "Location (km)": loc_km, "Elevation (m)": elev
        })
if st.session_state["peaks"]:
    st.dataframe(pd.DataFrame(st.session_state["peaks"]), use_container_width=True)
    if st.button("Clear All Peaks"):
        st.session_state["peaks"] = []
st.divider()

# 5. COSTS & PARAMS
st.subheader("Global & Cost Parameters")
colA, colB, colC, colD = st.columns(4)
dra_cost = colA.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
diesel_price = colB.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
grid_price = colC.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
time_horizon = colD.number_input("Scheduling Horizon (hours)", value=720, step=24)
min_v = st.number_input("Min velocity (m/s)", value=0.5)
max_v = st.number_input("Max velocity (m/s)", value=3.0)
st.divider()

# -------------------------------
# ---- NETWORK VISUALIZATION ----
# -------------------------------
st.subheader("Pipeline Network Visualization")
agraph_nodes = [Node(id=n["Name"], label=n["Name"], size=25, shape="circle") for n in st.session_state["nodes"]]
agraph_edges = [Edge(source=e["From Node"], target=e["To Node"], label=f"{e['Diameter (inches)']} in, {e['Length (km)']} km") for e in st.session_state["edges"]]
config = Config(width=900, height=400, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=False, node={"labelProperty":"label"}, link={"labelProperty":"label"}, staticGraph=False)
agraph(agraph_nodes, agraph_edges, config)
st.divider()

# -------------------------------
# ---- OPTIMIZATION AND OUTPUT ---
# -------------------------------
if st.button("Run Batch Network Optimization 🚀"):
    # Convert all inputs to backend format
    # Convert inch to meter
    nodes = st.session_state["nodes"]
    edges = []
    for e in st.session_state["edges"]:
        edges.append({
            "from_node": e["From Node"],
            "to_node": e["To Node"],
            "length_km": e["Length (km)"],
            "diameter_m": e["Diameter (inches)"] * 0.0254,
            "thickness_m": e["Thickness (inches)"] * 0.0254,
            "roughness": e["Roughness (m)"],
            "max_dr": e["Max DR (%)"]
        })
    pumps = []
    for p in st.session_state["pumps"]:
        pumps.append({
            "node_id": p["Station Name"],
            "branch_to": p["Branch To"],
            "power_type": p["Power Type"],
            "n_max": p["No. Pumps"],
            "min_rpm": p["Min RPM"],
            "max_rpm": p["Max RPM"],
            "sfc": p["SFC (Diesel)"],
            "grid_rate": p["Grid Rate (INR/kWh)"],
            # For demo: upload the CSV data as string, or parse as needed
            "head_curve_csv": p["Head Curve CSV"],
            "eff_curve_csv": p["Efficiency Curve CSV"],
            # Add default pump curve coefficients if not available
            "A": -0.0002, "B": 0.25, "C": 40, "P": -1e-8, "Q": 5e-6, "R": -0.0008, "S": 0.17, "T": 55
        })
    # Peaks: parse segment string to from/to
    peaks = {}
    for pk in st.session_state["peaks"]:
        seg_str = pk["Segment"]
        from_node, to_node = seg_str.split(" → ")
        seg_id = f"{from_node}_{to_node}"
        if seg_id not in peaks:
            peaks[seg_id] = []
        peaks[seg_id].append({"location_km": pk["Location (km)"], "elevation_m": pk["Elevation (m)"]})
    # Demands
    demands = {n["Name"]: n["Monthly Demand (m³)"] for n in nodes if n["Monthly Demand (m³)"] > 0}
    # Call backend
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
