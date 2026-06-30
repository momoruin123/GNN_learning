import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
import os

def build_mtsp_model(base, patrol_points, num_uav, max_energy=300,energy_rate=1):
    """
    Build MTSP MILP model
    
    Args:
        base: base location, shape (2,)
        patrol_points: patrol location, shape (n, 2)
        num_uav: number of uav
        max_energy: maximum energy of uav
        energy_rate: evergy cost rate
    
    Returns:
        model: Gurobi model
        data: 
    """
    points = np.vstack([base, patrol_points])
    n = len(points)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(points[i] - points[j])

    # ---------- model ----------
    model = gp.Model("MTSP")

    x = model.addVars(num_uav, n, n, vtype=GRB.BINARY, name='x')
    u = model.addVars(num_uav, n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name='u')
    t = model.addVar(lb=0, name="makespan")
    e = model.addVars(num_uav, n, vtype=GRB.CONTINUOUS, lb=0, ub=max_energy, name='e')

    # ---------- constratins ----------
    # flow balance constraints
    for d in range(num_uav):
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[d, i, j] for j in range(n)) == 
                gp.quicksum(x[d, j, i] for j in range(n)),
                name=f"flow_balance_{d}_{i}"
            )

    # visited once constraints
    for i in range(1, n):
        model.addConstr(
            gp.quicksum(x[d, i, j] for d in range(num_uav) for j in range(n) if j != i) == 1,
            name=f"visited_once_{i}"
        )

    for d in range(num_uav):
        for i in range(n):
            model.addConstr(x[d, i, i] == 0)

    for d in range(num_uav):
        model.addConstr(
            gp.quicksum(x[d, 0, j] for j in range(1, n)) == 1,
            name=f"start_{d}"
        )

    for d in range(num_uav):
        model.addConstr(
            gp.quicksum(x[d, i, 0] for i in range(1, n)) == 1,
            name=f"end_{d}"
        )

    # 7. MTZ Sub-loop elimination
    for d in range(num_uav):
        model.addConstr(u[d, 0] == 0)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(
                        u[d, i] - u[d, j] + n * x[d, i, j] <= n - 1,
                        name=f"mtz_{d}_{i}_{j}"
                    )

    # 8. Makespan 约束
    for d in range(num_uav):
        model.addConstr(
            gp.quicksum(dist[i, j] * x[d, i, j] for i in range(n) for j in range(n) if i != j) <= t,
            name=f"time_limit_{d}"
        )

    for d in range(num_uav):
        model.addConstr(e[d, 0] == max_energy,
                        name='base_energy')

    # 9. 能量消耗与续航严格约束
    for d in range(num_uav):
        for i in range(n):
            for j in range(n):
                if i != j:
                    if j != 0:
                        model.addGenConstrIndicator(
                            x[d, i, j], True,
                            e[d, j] == e[d, i] - dist[i][j] * energy_rate,
                            name=f"energy_to_point_{d}_{i}_{j}"
                        )
                    else:
                        model.addGenConstrIndicator(
                            x[d, i, 0], True,
                            e[d, i] - dist[i][0] * energy_rate >= 0,
                            name=f"energy_to_base_{d}_{i}"
                        )
    # ---------- 目标 ----------
    model.setObjective(t, GRB.MINIMIZE)

    # save mate data
    data = {
        'points': points,
        'dist': dist,
        'n': n,
        'num_uav': num_uav,
        'max_energy': max_energy,
        'base': base,
        'patrol_points': patrol_points,
        'vars': x,
    }
    
    return model, data