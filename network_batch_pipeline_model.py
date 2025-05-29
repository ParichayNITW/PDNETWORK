import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import pi, log10

def inch_to_m(val):
    # Utility: Convert inch to meter
    try:
        return float(val) * 0.0254
    except:
        return 0.0

def solve_batch_pipeline(
    stations, segments, pumps, peaks, demands, time_horizon_hours,
    dra_cost_per_litre, diesel_price, grid_price,
    min_velocity=0.5, max_velocity=3.0
):
    # Build node/edge dictionaries
    nodes = [n['Name'] for n in stations if n['Name']]
    node_elev = {n['Name']: float(n['Elevation (m)']) for n in stations if n['Name']}
    node_rho  = {n['Name']: float(n['Density (kg/m³)']) for n in stations if n['Name']}
    node_visc = {n['Name']: float(n['Viscosity (cSt)']) for n in stations if n['Name']}
    node_demand = {n['Name']: float(n['Monthly Demand (m³)']) for n in stations if n['Name'] and not pd.isna(n['Monthly Demand (m³)'])}

    # Assign edge IDs
    edge_ids = [f"E{i+1}" for i in range(len(segments))]
    edge_from = {eid: seg["From"] for eid, seg in zip(edge_ids, segments)}
    edge_to   = {eid: seg["To"] for eid, seg in zip(edge_ids, segments)}
    edge_len  = {eid: float(seg["Length (km)"])*1000 for eid, seg in zip(edge_ids, segments)}
    edge_diam = {eid: inch_to_m(seg["Diameter (inch)"]) for eid, seg in zip(edge_ids, segments)}
    edge_thic = {eid: inch_to_m(seg["Thickness (inch)"]) for eid, seg in zip(edge_ids, segments)}
    edge_maxdr= {eid: float(seg["Max DR (%)"]) for eid, seg in zip(edge_ids, segments)}
    edge_rough= {eid: float(seg["Roughness (m)"]) for eid, seg in zip(edge_ids, segments)}

    # Parse pumps
    pump_ids = [f"P{i+1}" for i in range(len(pumps))]
    pump_node = {pid: p["Station"] for pid, p in zip(pump_ids, pumps)}
    pump_branch = {pid: p["Branch To"] for pid, p in zip(pump_ids, pumps)}
    pump_ptype = {pid: p["Power Type"] for pid, p in zip(pump_ids, pumps)}
    pump_nmax  = {pid: int(p["No. Pumps"]) for pid, p in zip(pump_ids, pumps)}
    pump_minrpm= {pid: int(p["Min RPM"]) for pid, p in zip(pump_ids, pumps)}
    pump_maxrpm= {pid: int(p["Max RPM"]) for pid, p in zip(pump_ids, pumps)}
    pump_sfc   = {pid: float(p["SFC (Diesel)"]) for pid, p in zip(pump_ids, pumps)}
    pump_gridr = {pid: float(p["Grid Rate (INR/kWh)"]) for pid, p in zip(pump_ids, pumps)}
    # Default curves: You may extend this to load uploaded CSVs if needed!
    # For now use some example coefficients
    pump_head_curve = {pid: (-0.0002, 0.25, 40) for pid in pump_ids} # Quadratic: A, B, C
    pump_eff_curve = {pid: (-1e-8, 5e-6, -0.0008, 0.17, 55) for pid in pump_ids} # 4th degree

    # Peaks: {edge_id: [{location_km, elevation_m}, ...]}
    edge_peaks = {}
    if peaks and isinstance(peaks, list):
        for p in peaks:
            eid = p.get("Edge ID", "")
            if eid in edge_ids:
                edge_peaks.setdefault(eid, []).append({
                    "location_km": float(p.get("Location (km)", 0)),
                    "elevation_m": float(p.get("Elevation (m)", 0))
                })

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, time_horizon_hours)
    m.N = pyo.Set(initialize=nodes)
    m.E = pyo.Set(initialize=edge_ids)
    m.PUMP = pyo.Set(initialize=pump_ids)

    # Parameters
    m.dra_cost = pyo.Param(initialize=dra_cost_per_litre)
    m.diesel_price = pyo.Param(initialize=diesel_price)
    m.grid_price = pyo.Param(initialize=grid_price)
    m.min_v = pyo.Param(initialize=min_velocity)
    m.max_v = pyo.Param(initialize=max_velocity)

    demand_nodes = set(node_demand.keys())
    m.demand_nodes = pyo.Set(initialize=demand_nodes)
    m.demand_vol = pyo.Param(m.demand_nodes, initialize=node_demand)

    # Variables
    m.flow = pyo.Var(m.E, m.T, domain=pyo.NonNegativeReals)
    def dra_bounds(m, e, t): return (0, edge_maxdr[e])
    m.dra = pyo.Var(m.E, m.T, domain=pyo.NonNegativeReals, bounds=dra_bounds)
    m.rh = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)

    def n_bounds(m, p, t): return (0, pump_nmax[p])
    m.num_pumps = pyo.Var(m.PUMP, m.T, domain=pyo.NonNegativeIntegers, bounds=n_bounds)
    def rpm_bounds(m, p, t): return (0, pump_maxrpm[p])
    m.pump_rpm = pyo.Var(m.PUMP, m.T, domain=pyo.NonNegativeReals, bounds=rpm_bounds)
    m.pump_on = pyo.Var(m.PUMP, m.T, domain=pyo.Binary)

    # Continuity at each node/hour
    def flow_continuity_rule(m, n, t):
        inflow = sum(m.flow[e, t] for e in m.E if edge_to[e] == n)
        outflow = sum(m.flow[e, t] for e in m.E if edge_from[e] == n)
        return inflow == outflow
    m.flow_balance = pyo.Constraint(m.N, m.T, rule=flow_continuity_rule)

    # Demand fulfillment (over horizon)
    def demand_satisfaction_rule(m, n):
        inflow = sum(m.flow[e, t] for e in m.E for t in m.T if edge_to[e] == n)
        outflow = sum(m.flow[e, t] for e in m.E for t in m.T if edge_from[e] == n)
        net = inflow - outflow
        return net == m.demand_vol[n]
    m.demand_fulfillment = pyo.Constraint(m.demand_nodes, rule=demand_satisfaction_rule)

    # Pump logic: OFF => RPM and num_pumps zero
    def pump_rpm_status(m, p, t):
        return m.pump_rpm[p, t] <= m.pump_on[p, t] * pump_maxrpm[p]
    m.pump_rpm_status = pyo.Constraint(m.PUMP, m.T, rule=pump_rpm_status)
    def pump_num_on(m, p, t):
        return m.num_pumps[p, t] <= m.pump_on[p, t] * pump_nmax[p]
    m.pump_num_on = pyo.Constraint(m.PUMP, m.T, rule=pump_num_on)

    # Velocity limits
    def velocity_limit(m, e, t):
        d = edge_diam[e]
        area = pi * d**2 / 4
        v = m.flow[e, t] / 3600.0 / area
        return pyo.inequality(m.min_v, v, m.max_v)
    m.velocity_limits = pyo.Constraint(m.E, m.T, rule=velocity_limit)

    # Head balance, SDH (including peaks)
    g = 9.81
    def sdh_head_rule(m, e, t):
        from_n = edge_from[e]
        to_n = edge_to[e]
        d = edge_diam[e]
        rough = edge_rough[e]
        L = edge_len[e]
        rho = node_rho[from_n]
        kv = node_visc[from_n]
        Q = m.flow[e, t]
        area = pi * d**2 / 4
        v = Q / 3600.0 / area
        Re = v * d / (kv * 1e-6) if kv > 0 else 1e6
        if Re < 4000:
            f = 64 / max(Re, 1e-6)
        else:
            arg = rough/d/3.7 + 5.74/(Re**0.9)
            f = 0.25 / (log10(arg)**2) if arg > 0 else 0.015
        DR_frac = m.dra[e, t]/100.0
        loss = f * (L/d) * (v**2/(2*g)) * (1-DR_frac)
        # Peaks logic
        if e in edge_peaks:
            for pk in edge_peaks[e]:
                pk_loss = f * ((pk['location_km']*1000)/d) * (v**2/(2*g)) * (1-DR_frac)
                loss = max(loss, (pk['elevation_m'] - node_elev[from_n]) + pk_loss + 50)
        # Pump head
        pump_head = 0
        for pid in pump_ids:
            if pump_node[pid] == from_n and pump_branch[pid] == to_n:
                a, b, c = pump_head_curve[pid]
                pump_rpm = m.pump_rpm[pid, t]
                dol = pump_maxrpm[pid]
                H = (a*Q**2 + b*Q + c)*(pump_rpm/dol)**2 if dol > 0 else 0
                n_pump = m.num_pumps[pid, t]
                pump_head += H * n_pump
        return m.rh[from_n, t] + pump_head >= m.rh[to_n, t] + loss
    m.head_balance = pyo.Constraint(m.E, m.T, rule=sdh_head_rule)

    # Objective: total cost (power/fuel + DRA) over all hours
    def total_cost_rule(m):
        total_cost = 0
        for pid in pump_ids:
            node_id = pump_node[pid]
            power_type = pump_ptype[pid]
            a, b, c = pump_head_curve[pid]
            P, Qc, R, S, T = pump_eff_curve[pid]
            dol = pump_maxrpm[pid]
            sfc = pump_sfc[pid]
            rate = pump_gridr[pid]
            for t in range(1, time_horizon_hours+1):
                edges_out = [e for e in edge_ids if edge_from[e] == node_id and edge_to[e] == pump_branch[pid]]
                if not edges_out: continue
                eid = edges_out[0]
                Q = m.flow[eid, t]
                pump_rpm = m.pump_rpm[pid, t]
                n_pump = m.num_pumps[pid, t]
                H = (a*Q**2 + b*Q + c)*(pump_rpm/dol)**2 if dol > 0 else 0
                Qe = Q * dol/pump_rpm if pump_rpm > 0 else Q
                eff = (P*Qe**4 + Qc*Qe**3 + R*Qe**2 + S*Qe + T)/100.0 if pump_rpm > 0 else 0.5
                eff = max(0.05, eff)
                rho_val = node_rho[node_id]
                pwr_kW = (rho_val * Q * 9.81 * H * n_pump)/(3600.0 * 1000.0 * eff * 0.95)
                if power_type.lower() == "grid":
                    cost = pwr_kW * grid_price
                else:
                    fuel_per_kWh = (sfc*1.34102)/820.0
                    cost = pwr_kW * fuel_per_kWh * diesel_price
                total_cost += cost
        # DRA cost
        for e in edge_ids:
            for t in range(1, time_horizon_hours+1):
                dra_ppm = m.dra[e, t]
                Q = m.flow[e, t]
                dra_vol = dra_ppm * Q * 1000.0 / 1e6
                dra_cost = dra_vol * dra_cost_per_litre
                total_cost += dra_cost
        return total_cost
    m.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # ---- SOLVE ----
    results = SolverManagerFactory('neos').solve(m, solver='bonmin', tee=True)
    m.solutions.load_from(results)

    # ---- EXTRACT RESULTS ----
    results_dict = {
        "flow": {(e, t): pyo.value(m.flow[e, t]) for e in m.E for t in m.T},
        "dra": {(e, t): pyo.value(m.dra[e, t]) for e in m.E for t in m.T},
        "residual_head": {(n, t): pyo.value(m.rh[n, t]) for n in m.N for t in m.T},
        "pump_on": {(p, t): int(pyo.value(m.pump_on[p, t])) for p in m.PUMP for t in m.T},
        "pump_rpm": {(p, t): pyo.value(m.pump_rpm[p, t]) for p in m.PUMP for t in m.T},
        "num_pumps": {(p, t): int(pyo.value(m.num_pumps[p, t])) for p in m.PUMP for t in m.T},
        "total_cost": pyo.value(m.total_cost)
    }
    return results_dict
