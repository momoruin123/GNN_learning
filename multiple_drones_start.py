"""
对比：冷启动 vs 手动 MIPStart（固定实例）
"""
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ========== 固定数据（与 multiple_drones.py 完全一致）==========
BASE = np.array([0.0, 0.0])
PATROL_POINTS = np.array([
    [51.0, 92.0], [14.0, 71.0], [60.0, 20.0], [82.0, 86.0], [74.0, 74.0],
    [87.0, 99.0], [23.0, 2.0],  [21.0, 52.0], [1.0, 87.0],  [29.0, 37.0],
])
NUM_UAV = 3
MAX_ENERGY = 300
ENERGY_RATE = 1


def build_model():
    """构建 MTSP 模型（与 multiple_drones.py 完全一致）"""
    points = np.vstack([BASE, PATROL_POINTS])
    n = len(points)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(points[i] - points[j])

    model = gp.Model("MTSP")

    x = model.addVars(NUM_UAV, n, n, vtype=GRB.BINARY, name='x')
    u = model.addVars(NUM_UAV, n, vtype=GRB.CONTINUOUS, lb=0, ub=n - 1, name='u')
    t = model.addVar(lb=0, name="makespan")
    e = model.addVars(NUM_UAV, n, vtype=GRB.CONTINUOUS, lb=0, ub=MAX_ENERGY, name='e')

    # flow balance
    for d in range(NUM_UAV):
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[d, i, j] for j in range(n)) ==
                gp.quicksum(x[d, j, i] for j in range(n)),
                name=f"flow_balance_{d}_{i}"
            )

    # visited once
    for i in range(1, n):
        model.addConstr(
            gp.quicksum(x[d, i, j] for d in range(NUM_UAV) for j in range(n) if j != i) == 1,
            name=f"visited_once_{i}"
        )

    for d in range(NUM_UAV):
        for i in range(n):
            model.addConstr(x[d, i, i] == 0)

    # start / end
    for d in range(NUM_UAV):
        model.addConstr(
            gp.quicksum(x[d, 0, j] for j in range(1, n)) == 1, name=f"start_{d}")
        model.addConstr(
            gp.quicksum(x[d, i, 0] for i in range(1, n)) == 1, name=f"end_{d}")

    # MTZ
    for d in range(NUM_UAV):
        model.addConstr(u[d, 0] == 0)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(
                        u[d, i] - u[d, j] + n * x[d, i, j] <= n - 1,
                        name=f"mtz_{d}_{i}_{j}"
                    )

    # makespan
    for d in range(NUM_UAV):
        model.addConstr(
            gp.quicksum(dist[i, j] * x[d, i, j] for i in range(n) for j in range(n) if i != j) <= t,
            name=f"makespan_{d}"
        )

    # energy
    for d in range(NUM_UAV):
        model.addConstr(e[d, 0] == MAX_ENERGY)
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

    model.setObjective(t, GRB.MINIMIZE)
    return model, n, dist


# ========== 冷启动求解（不设 Start）==========
print("=" * 50)
print("【冷启动】无 MIPStart")
print("=" * 50)

model_cold, n, dist = build_model()
model_cold.setParam('OutputFlag', 0)

t0 = time.time()
model_cold.optimize()
t_cold = time.time() - t0

cold_obj = model_cold.ObjVal if model_cold.SolCount > 0 else None
print(f"状态: {model_cold.Status}  (2=OPTIMAL, 9=TIME_LIMIT)")
print(f"目标: {cold_obj:.2f}" if cold_obj else "目标: N/A")
print(f"耗时: {t_cold:.2f}s")
model_cold.dispose()

# ========== 手动 MIPStart 求解 ==========
print("\n" + "=" * 50)
print("【手动 MIPStart】随机初始猜测")
print("=" * 50)

model_warm, n, dist = build_model()
model_warm.setParam('OutputFlag', 0)

# 对所有 x 变量随机设 Start（5% 概率为 1）
import random
random.seed(42)
num_set = 0
for v in model_warm.getVars():
    if v.VarName.startswith('x['):
        val = 1 if random.random() < 0.05 else 0
        v.Start = val
        if val == 1:
            num_set += 1

print(f"设置了 {num_set} 个变量 Start = 1")

t0 = time.time()
model_warm.optimize()
t_warm = time.time() - t0

warm_obj = model_warm.ObjVal if model_warm.SolCount > 0 else None
print(f"状态: {model_warm.Status}")
print(f"目标: {warm_obj:.2f}" if warm_obj else "目标: N/A")
print(f"耗时: {t_warm:.2f}s")
model_warm.dispose()

# ========== 汇总 ==========
print("\n" + "=" * 50)
print(f"{'':<20} {'冷启动':>12} {'MIPStart':>12}")
print("-" * 50)
print(f"{'耗时':<20} {t_cold:>10.2f}s {t_warm:>10.2f}s")

if cold_obj and warm_obj:
    print(f"{'目标值':<20} {cold_obj:>10.2f} {warm_obj:>10.2f}")
    delta = (t_warm - t_cold) / t_cold * 100
    print(f"{'时间差异':<20} {'':>10} {delta:>+9.1f}%")
