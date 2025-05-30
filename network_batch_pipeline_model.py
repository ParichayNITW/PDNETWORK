import pyomo.environ as pyo
from pyomo.opt import SolverManagerFactory
from math import pi, log10

def solve_batch_pipeline(
    nodes, edges, pumps, peaks, demands, time_horizon_hours,
    dra_cost_per_litre, diesel_price, grid_price,
    min_velocity=0.5, max_velocity=3.0
):
    m = pyo.ConcreteModel()

    # ---- SETS ----
    m.T = pyo.RangeSet(1, time_horizon_hours)
    m.N = pyo.Set(initialize=[n['Name'] for n in nodes])
    m.E = pyo.Set(initialize([(e['from_node'], e['to_node']) for e in edges])
                 )  # Edge as tuple (from, to)
    m.PUMP = pyo.Set(initialize=[(p['node_id'], p['branch_to']) for p in pumps])

    edge_map = {(e['from_node'], e['to_node']): e for e in edges}
    node_map = {n['Name']: n for n in nodes}
    pump_map = {(p['node_id'], p['branch_to']): p for p in pumps}

    # ---- PARAMETERS ----
    elev = {n['Name']: n['Elevation (m)'] for n in nodes}
    density = {n['Name']: n['Density (kg/m³)'] for n in nodes}
    viscosity = {n['Name']: n['Viscosity (cSt)'] for n in nodes}

    length = {k: edge_map[k]['length_km'] * 1000 for k in edge_map}
    diameter = {k: edge_map[k]['diameter_m'] for k in edge_map}
    thickness = {k: edge_map[k]['thickness_m'] for k in edge_map}
    roughness = {k: edge_map[k].get('roughness', 0.00004) for k in edge_map}
    max_dr = {k: edge_map[k].get('max_dr', 0.0) for k in edge_map}

    # DRA/cost
    m.dra_cost = pyo.Param(initialize=dra_cost_per_litre)
    m.diesel_price = pyo.Param(initialize=diesel_price)
    m.grid_price = pyo.Param(initialize=grid_price)
    m.min_v = pyo.Param(initialize=min_velocity)
    m.max_v = pyo.Param(initialize=max_velocity)

    # Demand
    m.demand_nodes = pyo.Set(initialize=set(demands.keys()))
    m.demand_vol = pyo.Param(m.demand_nodes, initialize=demands)
    # Peaks
    edge_peaks = peaks

    # ---- VARIABLES ----
    m.flow = pyo.Var(m.E, m.T, domain=pyo.NonNegativeReals)
    def dra_bounds(m, e, t): return (0, max_dr[e])
    m.dra = pyo.Var(m.E, m.T, domain=pyo.NonNegativeReals, bounds=dra_bounds)
    m.rh = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)

    # Pumps
    pump_max = {k: pump_map[k]['n_max'] for k in m.PUMP}
    pump_min_rpm = {k: pump_map[k]['min_rpm'] for k in m.PUMP}
    pump_max_rpm = {k: pump_map[k]['max_rpm'] for k in m.PUMP}
    def n_bounds(m, p, t): return (0, pump_max[p])
    m.num_pumps = pyo.Var(m.PUMP, m.T, domain=pyo.NonNegativeIntegers, bounds=n_bounds)
    def rpm_bounds(m, p, t): return (0, pump_max_rpm[p])
    m.pump_rpm = pyo.Var(m.PUMP, m.T, domain=pyo.NonNegativeReals, bounds=rpm_bounds)
    m.pump_on = pyo.Var(m.PUMP, m.T, domain=pyo.Binary)

    # ---- CONSTRAINTS ----
    # Flow continuity
    def flow_continuity_rule(m, n, t):
        inflow = sum(m.flow[e, t] for e in m.E if e[1] == n)
        outflow = sum(m.flow[e, t] for e in m.E if e[0] == n)
        return inflow == outflow
    m.flow_balance = pyo.Constraint(m.N, m.T, rule=flow_continuity_rule)

    # Demand fulfillment (over horizon)
    def demand_satisfaction_rule(m, n):
        inflow = sum(m.flow[e, t] for e in m.E for t in m.T if e[1] == n)
        outflow = sum(m.flow[e, t] for e in m.E for t in m.T if e[0] == n)
        net = inflow - outflow
        return net == m.demand_vol[n]
    m.demand_fulfillment = pyo.Constraint(m.demand_nodes, rule=demand_satisfaction_rule)

    # Pump ON/OFF
    def pump_rpm_status(m, p, t):
        return m.pump_rpm[p, t] <= m.pump_on[p, t] * pump_max_rpm[p]
    m.pump_rpm_status = pyo.Constraint(m.PUMP, m.T, rule=pump_rpm_status)
    def pump_num_on(m, p, t):
        return m.num_pumps[p, t] <= m.pump_on[p, t] * pump_max[p]
    m.pump_num_on = pyo.Constraint(m.PUMP, m.T, rule=pump_num_on)

    # Velocity limits
    def velocity_limit(m, e, t):
        d = diameter[e]
        area = pi * d**2 / 4
        v = m.flow[e, t] / 3600.0 / area
        return pyo.inequality(m.min_v, v, m.max_v)
    m.velocity_limits = pyo.Constraint(m.E, m.T, rule=velocity_limit)

    # SDH, Peak and Head constraint
    g = 9.81
    def sdh_head_rule(m, e, t):
        from_n, to_n = e
        d = diameter[e]
        rough = roughness[e]
        L = length[e]
        rho = density[from_n]
        kv = viscosity[from_n]
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
        eid = f"{from_n}_{to_n}"
        if eid in edge_peaks:
            for pk in edge_peaks[eid]:
                pk_loss = f * ((pk['location_km']*1000)/d) * (v**2/(2*g)) * (1-DR_frac)
                loss = max(loss, (pk['elevation_m'] - elev[from_n]) + pk_loss + 50)
        # Pump head
        pump_head = 0
        for p in m.PUMP:
            if p[0] == from_n and p[1] == to_n:
                pump = pump_map[p]
                a, b, c = pump['A'], pump['B'], pump['C']
                pump_rpm = m.pump_rpm[p, t]
                dol = pump['max_rpm']
                H = (a*Q**2 + b*Q + c)*(pump_rpm/dol)**2 if dol > 0 else 0
                n_pump = m.num_pumps[p, t]
                pump_head += H * n_pump
        return m.rh[from_n, t] + pump_head >= m.rh[to_n, t] + loss
    m.head_balance = pyo.Constraint(m.E, m.T, rule=sdh_head_rule)

    # ---- OBJECTIVE: Minimize cost ----
    def total_cost_rule(m):
        total_cost = 0
        for p in m.PUMP:
            pump = pump_map[p]
            node_id = p[0]
            power_type = pump['power_type']
            a, b, c = pump['A'], pump['B'], pump['C']
            P, Qc, R, S, T = pump['P'], pump['Q'], pump['R'], pump['S'], pump['T']
            dol = pump['max_rpm']
            sfc = pump.get('sfc', 0)
            rate = pump.get('grid_rate', 0)
            for t in m.T:
                eid = (p[0], p[1])
                if eid not in m.E:
                    continue
                Q = m.flow[eid, t]
                pump_rpm = m.pump_rpm[p, t]
                n_pump = m.num_pumps[p, t]
                H = (a*Q**2 + b*Q + c)*(pump_rpm/dol)**2 if dol > 0 else 0
                Qe = Q * dol/pump_rpm if pump_rpm > 0 else Q
                eff = (P*Qe**4 + Qc*Qe**3 + R*Qe**2 + S*Qe + T)/100.0 if pump_rpm > 0 else 0.5
                eff = max(0.05, eff)
                rho_val = density[node_id]
                pwr_kW = (rho_val * Q * 9.81 * H * n_pump)/(3600.0 * 1000.0 * eff * 0.95)
                if power_type.lower() == "grid":
                    cost = pwr_kW * grid_price
                else:
                    fuel_per_kWh = (sfc*1.34102)/820.0
                    cost = pwr_kW * fuel_per_kWh * diesel_price
                total_cost += cost
        # DRA cost
        for e in m.E:
            for t in m.T:
                dra_ppm = m.dra[e, t]
                Q = m.flow[e, t]
                dra_vol = dra_ppm * Q * 1000.0 / 1e6  # L/hr
                dra_cost = dra_vol * dra_cost_per_litre
                total_cost += dra_cost
        return total_cost
    m.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # ---- SOLVE ----
    solver_manager = SolverManagerFactory('neos')
    results = solver_manager.solve(m, solver='bonmin', tee=True)
    m.solutions.load_from(results)

    # ---- EXTRACT RESULTS ----
    results_dict = {
        "flow": {(f"{e[0]}→{e[1]}", t): pyo.value(m.flow[e, t]) for e in m.E for t in m.T},
        "dra": {(f"{e[0]}→{e[1]}", t): pyo.value(m.dra[e, t]) for e in m.E for t in m.T},
        "residual_head": {(n, t): pyo.value(m.rh[n, t]) for n in m.N for t in m.T},
        "pump_on": {(p[0], t): int(pyo.value(m.pump_on[p, t])) for p in m.PUMP for t in m.T},
        "pump_rpm": {(p[0], t): pyo.value(m.pump_rpm[p, t]) for p in m.PUMP for t in m.T},
        "num_pumps": {(p[0], t): int(pyo.value(m.num_pumps[p, t])) for p in m.PUMP for t in m.T},
        "total_cost": pyo.value(m.total_cost)
    }
    return results_dict
