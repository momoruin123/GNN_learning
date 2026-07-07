"""
对比：冷启动 / 最优解 MIPStart / 烂解 MIPStart（固定实例）

Gurobi MIPStart 用法：直接给变量设 v.Start = 1.0 或 0.0，然后正常 optimize() 即可。
不需要额外参数，Gurobi 会自动读 Start 值作为初始猜测。
"""
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ========== 固定数据 ==========
BASE = np.array([0.0, 0.0])
PATROL_POINTS = np.array([
    [51.0, 92.0], [14.0, 71.0], [60.0, 20.0], [82.0, 86.0], [74.0, 74.0],
    [87.0, 99.0], [23.0, 2.0],  [21.0, 52.0], [1.0, 87.0],  [29.0, 37.0],
    [45.0, 55.0], [68.0, 10.0], [90.0, 45.0], [33.0, 80.0], [55.0, 15.0],
    [77.0, 66.0], [12.0, 33.0], [40.0, 70.0], [88.0, 12.0], [95.0, 70.0],
])
NUM_UAV = 4
MAX_ENERGY = 600

points = np.vstack([BASE, PATROL_POINTS])
n = len(points)
dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist[i, j] = np.linalg.norm(points[i] - points[j])


def build_model():
    m = gp.Model("MTSP")
    x = m.addVars(NUM_UAV, n, n, vtype=GRB.BINARY, name='x')
    u = m.addVars(NUM_UAV, n, vtype=GRB.CONTINUOUS, lb=0, ub=n - 1, name='u')
    t = m.addVar(lb=0, name="makespan")
    e = m.addVars(NUM_UAV, n, vtype=GRB.CONTINUOUS, lb=0, ub=MAX_ENERGY, name='e')

    for d in range(NUM_UAV):
        for i in range(n):
            m.addConstr(gp.quicksum(x[d, i, j] for j in range(n)) ==
                        gp.quicksum(x[d, j, i] for j in range(n)))

    for i in range(1, n):
        m.addConstr(gp.quicksum(x[d, i, j] for d in range(NUM_UAV)
                                for j in range(n) if j != i) == 1)

    for d in range(NUM_UAV):
        for i in range(n):
            m.addConstr(x[d, i, i] == 0)
        m.addConstr(gp.quicksum(x[d, 0, j] for j in range(1, n)) == 1)
        m.addConstr(gp.quicksum(x[d, i, 0] for i in range(1, n)) == 1)

    for d in range(NUM_UAV):
        m.addConstr(u[d, 0] == 0)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    m.addConstr(u[d, i] - u[d, j] + n * x[d, i, j] <= n - 1)

    for d in range(NUM_UAV):
        m.addConstr(gp.quicksum(dist[i, j] * x[d, i, j]
                                for i in range(n) for j in range(n) if i != j) <= t)

    for d in range(NUM_UAV):
        m.addConstr(e[d, 0] == MAX_ENERGY)
    for d in range(NUM_UAV):
        for i in range(n):
            for j in range(n):
                if i != j:
                    if j != 0:
                        m.addGenConstrIndicator(x[d, i, j], True,
                            e[d, j] == e[d, i] - dist[i][j])
                    else:
                        m.addGenConstrIndicator(x[d, i, 0], True,
                            e[d, i] - dist[i][0] >= 0)

    m.setObjective(t, GRB.MINIMIZE)
    return m, x


def do_solve(m, label):
    m.setParam('TimeLimit', 120)
    m.setParam('OutputFlag', 0)
    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0
    obj = m.ObjVal if m.SolCount > 0 else None
    print(f"【{label}】")
    print(f"  状态: {m.Status}  (2=OPTIMAL, 9=TIME_LIMIT)")
    print(f"  目标: {obj:.2f}" if obj else "  目标: N/A")
    print(f"  耗时: {elapsed:.2f}s")
    return elapsed, obj


# ===== 1. 冷启动 =====
print("=" * 55)
m1, x1 = build_model()
t1, obj1 = do_solve(m1, "冷启动（无 MIPStart）")

# 取出最优解中所有 x[d,i,j] 的值
best = {}  # (d,i,j) -> 0/1
if m1.SolCount > 0:
    for d in range(NUM_UAV):
        for i in range(n):
            for j in range(n):
                if i != j:
                    best[(d, i, j)] = int(round(x1[d, i, j].X))
m1.dispose()

# ===== 2. 最优解作为 MIPStart =====
print()
m2, x2 = build_model()
n_set = 0
for (d, i, j), val in best.items():
    if val == 1:
        x2[d, i, j].Start = 1.0
        n_set += 1
print(f"【最优解作 MIPStart】设置了 {n_set} 个变量 = 1（就是最优解本身）")
t2, obj2 = do_solve(m2, "最优解作 MIPStart")
m2.dispose()

# ===== 3. 烂解 MIPStart =====
# d=0 按顺序跑全部巡逻点; d=1,d=2 各跑离基地最近的一个点
print()
m3, x3 = build_model()

# 先把所有 x 设为 0
for d in range(NUM_UAV):
    for i in range(n):
        for j in range(n):
            if i != j:
                x3[d, i, j].Start = 0.0

# d=0: 0→1→2→...→n-1→0
n_set = 0
for k in range(1, n):
    x3[0, k - 1, k].Start = 1.0
    n_set += 1
x3[0, n - 1, 0].Start = 1.0
n_set += 1

# d=1,d=2: 捡 d=0 路线最远一段的两端，故意制造超大 makespan
# d=0 已经覆盖了全部巡逻点，d=1,d=2 再重复指过去（Gurobi 会自己修正）
dist_from_base = sorted([(i, dist[0][i]) for i in range(1, n)], key=lambda v: v[1])
far_node = dist_from_base[-1][0]   # 离基地最远的点
far2_node = dist_from_base[-2][0]  # 次远的点
for d, node in [(1, far_node), (2, far2_node)]:
    x3[d, 0, node].Start = 1.0
    x3[d, node, 0].Start = 1.0
    n_set += 2
print(f"【烂解 MIPStart】设置了 {n_set} 个变量 = 1")
print(f"  d=0: 基地 → P1 → P2 → ... → P{len(PATROL_POINTS)} → 基地")
print(f"  d=1: 基地 → P{far_node}（最远点） → 基地")
print(f"  d=2: 基地 → P{far2_node}（次远点） → 基地")
t3, obj3 = do_solve(m3, "烂解作 MIPStart")
m3.dispose()

# ===== 汇总 =====
print("\n" + "=" * 55)
print(f"{'':<22} {'耗时':>10} {'目标值':>10}")
print("-" * 55)
for label, t, obj in [("冷启动", t1, obj1),
                       ("最优解 MIPStart", t2, obj2),
                       ("烂解 MIPStart", t3, obj3)]:
    obj_str = f"{obj:.2f}" if obj else "N/A"
    print(f"{label:<22} {t:>8.2f}s {obj_str:>10}")
print("-" * 55)
if t1:
    print(f"最优解 vs 冷启动: {(t2 - t1) / t1 * 100:+.1f}%")
    print(f"烂解   vs 冷启动: {(t3 - t1) / t1 * 100:+.1f}%")
