import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
import json
import io

st.set_page_config(page_title="Pipeline Optima™ Network Batch Scheduler", layout="wide")

st.markdown(
    "<h1 style='text-align:center;font-size:2.5rem;font-weight:700;color:#232733;margin-bottom:0.25em;margin-top:0.01em;'>Pipeline Optima™ Network Batch Scheduler</h1>",
    unsafe_allow_html=True
)

st.markdown("<hr style='margin-top:0.4em; margin-bottom:1em; border: 1px solid #e1e5ec;'>", unsafe_allow_html=True)

# -----------------------------
# Session state for all entities
# -----------------------------

if "stations" not in st.session_state:
    st.session_state["stations"] = []
if "segments" not in st.session_state:
    st.session_state["segments"] = []
if "pumps" not in st.session_state:
    st.session_state["pumps"] = []
if "peaks" not in st.session_state:
    st.session_state["peaks"] = []

# ---- Tabbed Input UI ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stations", "Pipe Segments", "Pumps", "Peaks", "Cost & Limits"])

# ------ 1. Stations ------
with tab1:
    st.subheader("Add Station / Demand Center")
    with st.form("station_form", clear_on_submit=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        s_name = col1.text_input("Station Name (unique)", max_chars=20)
        s_elev = col2.number_input("Elevation (m)", value=0.0, step=0.1)
        s_rho = col3.number_input("Density (kg/m³)", value=850.0, step=1.0)
        s_visc = col4.number_input("Viscosity (cSt)", value=10.0, step=0.1)
        s_demand = col5.number_input("Monthly Demand (m³)", value=0.0, step=100.0)
        add_station = st.form_submit_button("Add Station")
        if add_station and s_name and all(s_name != s["Name"] for s in st.session_state["stations"]):
            st.session_state["stations"].append({
                "Name": s_name, "Elevation (m)": s_elev, "Density (kg/m³)": s_rho,
                "Viscosity (cSt)": s_visc, "Monthly Demand (m³)": s_demand
            })
    if st.session_state["stations"]:
        st.dataframe(pd.DataFrame(st.session_state["stations"]), hide_index=True, use_container_width=True)
        if st.button("Clear All Stations"):
            st.session_state["stations"] = []

# ------ 2. Pipe Segments ------
with tab2:
    st.subheader("Add Pipe Segment (Between Stations)")
    station_names = [s["Name"] for s in st.session_state["stations"]]
    with st.form("segment_form", clear_on_submit=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        seg_from = c1.selectbox("From Station", station_names, key="seg_from")
        seg_to = c2.selectbox("To Station", station_names, key="seg_to")
        seg_len = c3.number_input("Length (km)", value=100.0, min_value=0.1, step=1.0)
        seg_dia_in = c4.number_input("Diameter (inch)", value=28.0, min_value=1.0, step=0.5)
        seg_thick_in = c5.number_input("Thickness (inch)", value=0.28, min_value=0.01, step=0.01)
        seg_maxdr = c6.number_input("Max DR (%)", value=40.0, min_value=0.0, max_value=99.9, step=1.0)
        seg_rough = st.number_input("Roughness (m)", value=0.00004, format="%.5f", step=0.00001)
        add_segment = st.form_submit_button("Add Segment")
        if add_segment and seg_from != seg_to:
            st.session_state["segments"].append({
                "From": seg_from, "To": seg_to, "Length (km)": seg_len,
                "Diameter (m)": seg_dia_in * 0.0254,   # Auto-convert inch to m
                "Thickness (m)": seg_thick_in * 0.0254,
                "Max DR (%)": seg_maxdr, "Roughness (m)": seg_rough
            })
    if st.session_state["segments"]:
        df_seg = pd.DataFrame(st.session_state["segments"])
        df_seg_disp = df_seg.copy()
        df_seg_disp["Diameter (inch)"] = (df_seg_disp["Diameter (m)"] / 0.0254).round(2)
        df_seg_disp["Thickness (inch)"] = (df_seg_disp["Thickness (m)"] / 0.0254).round(3)
        st.dataframe(df_seg_disp[["From", "To", "Length (km)", "Diameter (inch)", "Thickness (inch)", "Max DR (%)", "Roughness (m)"]], hide_index=True, use_container_width=True)
        if st.button("Clear All Segments"):
            st.session_state["segments"] = []

# ------ 3. Pumps ------
with tab3:
    st.subheader("Add Pumping Unit")
    branch_names = [s["Name"] for s in st.session_state["stations"]]
    with st.form("pump_form", clear_on_submit=True):
        p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
        stn_name = p1.selectbox("Station Name", station_names, key="pump_stn")
        branch_to = p2.selectbox("Branch To", branch_names, key="pump_branch_to")
        power_type = p3.selectbox("Power Type", ["Grid", "Diesel"], key="pump_ptype")
        n_pumps = p4.number_input("No. Pumps", min_value=1, value=1, step=1, key="pump_n")
        min_rpm = p5.number_input("Min RPM", min_value=0, value=1000, step=50, key="pump_minrpm")
        max_rpm = p6.number_input("Max RPM", min_value=0, value=1500, step=50, key="pump_maxrpm")
        sfc = p7.number_input("SFC (Diesel)", min_value=0.0, value=150.0, step=1.0, key="pump_sfc")
        grid_rate = st.number_input("Grid Rate (INR/kWh)", min_value=0.0, value=9.0, step=0.1, key="pump_grid")
        head_csv = st.file_uploader("Pump Head Curve CSV", type="csv", key="pump_head")
        eff_csv = st.file_uploader("Pump Efficiency Curve CSV", type="csv", key="pump_eff")
        add_pump = st.form_submit_button("Add Pump")
        if add_pump and stn_name and branch_to and stn_name != branch_to:
            st.session_state["pumps"].append({
                "Station Name": stn_name, "Branch To": branch_to, "Power Type": power_type,
                "No. Pumps": n_pumps, "Min RPM": min_rpm, "Max RPM": max_rpm,
                "SFC (Diesel)": sfc, "Grid Rate": grid_rate,
                "Head Curve CSV": head_csv, "Efficiency Curve CSV": eff_csv
            })
    if st.session_state["pumps"]:
        st.dataframe(pd.DataFrame([
            {k: (v.name if hasattr(v, 'name') else v) for k,v in pump.items()}
            for pump in st.session_state["pumps"]
        ]), hide_index=True, use_container_width=True)
        if st.button("Clear All Pumps"):
            st.session_state["pumps"] = []

# ------ 4. Peaks ------
with tab4:
    st.subheader("Add Elevation Peak (between 2 stations along a pipe segment)")
    seg_labels = [f"{s['From']} ➔ {s['To']}" for s in st.session_state["segments"]]
    with st.form("peak_form", clear_on_submit=True):
        if seg_labels:
            pk_seg = st.selectbox("Which Pipe Segment?", seg_labels, key="peak_seg")
            pk_loc = st.number_input("Location (km from 'From' station)", value=1.0, min_value=0.01, step=0.01)
            pk_elev = st.number_input("Elevation (m)", value=10.0, step=0.1)
            add_peak = st.form_submit_button("Add Peak")
            if add_peak:
                st.session_state["peaks"].append({
                    "Segment": pk_seg, "Location (km)": pk_loc, "Elevation (m)": pk_elev
                })
    if st.session_state["peaks"]:
        st.dataframe(pd.DataFrame(st.session_state["peaks"]), hide_index=True, use_container_width=True)
        if st.button("Clear All Peaks"):
            st.session_state["peaks"] = []

# ------ 5. Cost & Limits ------
with tab5:
    st.subheader("Set Global Cost & Operating Limits")
    c1, c2, c3 = st.columns(3)
    dra_cost = c1.number_input("DRA Cost (INR/L)", value=500.0, step=1.0)
    diesel_price = c2.number_input("Diesel Price (INR/L)", value=70.0, step=0.5)
    grid_price = c3.number_input("Grid Electricity (INR/kWh)", value=9.0, step=0.1)
    min_v = st.number_input("Min velocity (m/s)", value=0.5, step=0.1)
    max_v = st.number_input("Max velocity (m/s)", value=3.0, step=0.1)
    time_horizon = st.number_input("Scheduling Horizon (hours)", value=720, step=24)

# ------ Network Visualization (Interactive) ------
st.markdown("---")
st.subheader("Network Preview (Interactive)")
if st.session_state["stations"] and st.session_state["segments"]:
    # Build node/edge list for agraph
    nodelist = []
    for s in st.session_state["stations"]:
        nodelist.append(Node(id=s["Name"], label=s["Name"], size=35, color="#1976d2"))
    edgelist = []
    for seg in st.session_state["segments"]:
        edgelist.append(Edge(source=seg["From"], target=seg["To"], label=f"{(seg['Diameter (m)']/0.0254):.1f}''", color="#555"))
    config = Config(width=1200, height=500, directed=True,
        nodeHighlightBehavior=True, highlightColor="#F7A7A6",
        collapsible=True, node={'color': '#1976d2', 'size': 600, 'highlightStrokeColor': '#A00000'},
        link={'labelProperty': 'label', 'highlightColor': '#00F'},
        staticGraph=False
    )
    agraph(nodes=nodelist, edges=edgelist, config=config)
else:
    st.info("Add at least two stations and one segment to see the network.")

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 2em; font-size: 0.9em;'>&copy; 2025 Pipeline Optima™ v2.0. Developed by Parichay Das. All rights reserved.</div>",
    unsafe_allow_html=True
)
