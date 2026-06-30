import numpy as np
import gurobipy as gp
from gurobipy import GRB

BASE = np.array([0.0, 0.0])
PATROL_POINTS = np.array([
    [51.0, 92.0],
    [14.0, 71.0],
    [60.0, 20.0],
    [82.0, 86.0],
    [74.0, 74.0],
    [87.0, 99.0],
    [23.0, 2.0],
    [21.0, 52.0],
    [1.0, 87.0],
    [29.0, 37.0],
])
NUM_UAV = 3
MAX_ENERGY = 300
ENERGY_RATE = 1
BIG_M = MAX_ENERGY*2

points = np.vstack([BASE, PATROL_POINTS])
n = len(points)

dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist[i, j] = np.linalg.norm(points[i] - points[j])

# ---------- model ----------
model = gp.Model("MTSP")

x = model.addVars(NUM_UAV, n, n, vtype=GRB.BINARY, name='x')
u = model.addVars(NUM_UAV, n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name='u')
t = model.addVar(lb=0, name="makespan")
e = model.addVars(NUM_UAV, n, vtype=GRB.CONTINUOUS, lb=0, ub=MAX_ENERGY, name='e')

# ---------- 约束 ----------
# flow balance constraints
for d in range(NUM_UAV):
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[d, i, j] for j in range(n)) == 
            gp.quicksum(x[d, j, i] for j in range(n)),
            name=f"flow_balance_{d}_{i}"
        )

# visited once constraints
for i in range(1, n):
    model.addConstr(
        gp.quicksum(x[d, i, j] for d in range(NUM_UAV) for j in range(n) if j != i) == 1,
        name=f"visited_once_{i}"
    )

for d in range(NUM_UAV):
    for i in range(n):
        model.addConstr(x[d, i, i] == 0)

for d in range(NUM_UAV):
    model.addConstr(
        gp.quicksum(x[d, 0, j] for j in range(1, n)) == 1,
        name=f"start_{d}"
    )

for d in range(NUM_UAV):
    model.addConstr(
        gp.quicksum(x[d, i, 0] for i in range(1, n)) == 1,
        name=f"end_{d}"
    )

# 7. MTZ Sub-loop elimination
for d in range(NUM_UAV):
    model.addConstr(u[d, 0] == 0)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(
                    u[d, i] - u[d, j] + n * x[d, i, j] <= n - 1,
                    name=f"mtz_{d}_{i}_{j}"
                )

# 8. Makespan 约束
for d in range(NUM_UAV):
    model.addConstr(
        gp.quicksum(dist[i, j] * x[d, i, j] for i in range(n) for j in range(n) if i != j) <= t,
        name=f"time_limit_{d}"
    )

for d in range(NUM_UAV):
    model.addConstr(e[d, 0] == MAX_ENERGY)

# 9. 能量消耗与续航严格约束
for d in range(NUM_UAV):
    for i in range(n):
        for j in range(n):
            if i != j:
                if j != 0:
                    model.addGenConstrIndicator(
                        x[d, i, j], True,
                        e[d, j] == e[d, i] - dist[i][j] * ENERGY_RATE,
                        name=f"energy_to_point_{d}_{i}_{j}"
                    )
                else:
                    model.addGenConstrIndicator(
                        x[d, i, 0], True,
                        e[d, i] - dist[i][0] * ENERGY_RATE >= 0,
                        name=f"energy_to_base_{d}_{i}"
                    )
# ---------- 目标 ----------
model.setObjective(t, GRB.MINIMIZE)

model.optimize()

# --------- 输出 ----------
if model.status == GRB. OPTIMAL:
    print(f"\nMakespan: {t.X:.2f}\n")
    
    names = ['基地'] + [f'P{i}' for i in range(1, n)]
    
    for d in range(NUM_UAV):
        # 检查这架无人机是否真的出发了
        if sum(x[d, 0, j].X for j in range(1, n)) < 0.5:
            print(f"✈️ 无人机 {d}: 未使用")
            continue
        
        # 严格沿着 x[d, i, j] == 1 的轨迹寻找下一跳
        route = [0]
        cur = 0
        while True:
            next_node = None
            for j in range(n):
                if j != cur and x[d, cur, j].X > 0.5:
                    next_node = j
                    break
            
            if next_node is not None:
                route.append(next_node)
                cur = next_node
                if next_node == 0: # 重新回到基地，路线结束
                    break
            else:
                # 防御性容错：万一约束没管住出现断路（正常情况下不会执行到这）
                print(f"⚠️ 无人机 {d} 轨迹异常断开")
                break
        
        # 1. 打印路线
        print(f"无人机 {d}:")
        print("   " + " → ".join(names[i] for i in route))
        
        print("   电量:")
        for idx, node in enumerate(route):
            if node == 0 and idx == len(route) - 1:
                prev_node = route[idx-1]
                arrival_energy = e[d, prev_node].X - dist[prev_node, 0] * ENERGY_RATE
                print(f"     ➔ 回到 {names[node]}: 剩余电量 {arrival_energy:.2f}")
            else:
                print(f"     ➔ 到达 {names[node]}: 剩余电量 {e[d, node].X:.2f}")
        print("-" * 30)
        
else:
    print("无可行解")

    # 获取非零系数矩阵（稀疏矩阵格式）
model.update()
A = model.getA()  # 这是 scipy.sparse.csr_matrix316

vars = model.getVars()
cons = model.getConstrs()

print("debug")
