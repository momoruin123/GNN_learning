"""
对比三种 MIPStart：冷启动 / 最优解 / 烂解（单机跑全部）
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
    """构建 MTSP 模型"""
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

    for d in range(NUM_UAV):
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[d, i, j] for j in range(n)) ==
                gp.quicksum(x[d, j, i] for j in range(n)),
                name=f"flow_balance_{d}_{i}")

    for i in range(1, n):
        model.addConstr(
            gp.quicksum(x[d, i, j] for d in range(NUM_UAV) for j in range(n) if j != i) == 1,
            name=f"visited_once_{i}")

    for d in range(NUM_UAV):
        for i in range(n):
            model.addConstr(x[d, i, i] == 0)

    for d in range(NUM_UAV):
        model.addConstr(gp.quicksum(x[d, 0, j] for j in range(1, n)) == 1, name=f"start_{d}")
        model.addConstr(gp.quicksum(x[d, i, 0] for i in range(1, n)) == 1, name=f"end_{d}")

    for d in range(NUM_UAV):
        model.addConstr(u[d, 0] == 0)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(
                        u[d, i] - u[d, j] + n * x[d, i, j] <= n - 1,
                        name=f"mtz_{d}_{i}_{j}")

    for d in range(NUM_UAV):
        model.addConstr(
            gp.quicksum(dist[i, j] * x[d, i, j] for i in range(n) for j in range(n) if i != j) <= t,
            name=f"makespan_{d}")

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
                            name=f"energy_to_point_{d}_{i}_{j}")
                    else:
                        model.addGenConstrIndicator(
                            x[d, i, 0], True,
                            e[d, i] - dist[i][0] * ENERGY_RATE >= 0,
                            name=f"energy_to_base_{d}_{i}")

    model.setObjective(t, GRB.MINIMIZE)
    return model, n, dist


def solve(model, label, time_limit=120):
    """统一求解 + 计时"""
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    t0 = time.time()
    model.optimize()
    elapsed = time.time() - t0
    obj = model.ObjVal if model.SolCount > 0 else None
    print(f"【{label}】")
    print(f"  状态: {model.Status}  目标: {obj:.2f}" if obj else f"  状态: {model.Status}  目标: N/A")
    print(f"  耗时: {elapsed:.2f}s")
    return elapsed, obj, model.Status


# ============================================================
# 1. 冷启动（不设任何 Start）
# ============================================================
print("=" * 55)
model1, n, dist = build_model()
t1, obj1, st1 = solve(model1, "冷启动（无 MIPStart）")

# 保存最优解
best_x = {}
if model1.SolCount > 0:
    for v in model1.getVars():
        if v.VarName.startswith('x['):
            best_x[v.VarName] = int(round(v.X))
model1.dispose()

# ============================================================
# 2. 最优解作为 MIPStart
# ============================================================
print()
model2, n, dist = build_model()
n_set = 0
for v in model2.getVars():
    if v.VarName.startswith('x['):
        v.Start = best_x.get(v.VarName, 0)
        if v.Start == 1:
            n_set += 1
print(f"【最优解作 MIPStart】设置了 {n_set} 个变量 = 1（就是最优解本身）")
t2, obj2, st2 = solve(model2, "最优解作 MIPStart")
model2.dispose()

# ============================================================
# 3. 烂解：无人机 0 跑全部巡逻点，其余待命
# ============================================================
print()
model3, n, dist = build_model()

# 无人机 0：基地 → P1 → P2 → ... → P10 → 基地
patrol_indices = list(range(1, n))  # 1, 2, ..., 10
n_set = 0
for v in model3.getVars():
    if v.VarName.startswith('x['):
        parts = v.VarName.replace('x[', '').replace(']', '').split(',')
        d, i, j = int(parts[0]), int(parts[1]), int(parts[2])
        if d == 0:
            # 按顺序连接: 0→1, 1→2, 2→3, ...→0
            if (i == 0 and j == 1) or (i == j - 1 and j <= n - 1) or (i == n - 1 and j == 0):
                v.Start = 1.0
                n_set += 1
            else:
                pass  # GRB.UNDEFINED
        else:
            # 其余无人机直接从基地出发立即回基地
            if i == 0 and j == 0:
                pass  # 跳过自环
            elif i == 0 and j == 0:  # 这个条件不会触发
                pass

        # 无人机 1、2：基地出发立即返回（不做任何巡逻）
        # 但这在约束里是不允许的（start/end 各一次但必须离开基地）
        # 所以我们让它们任意飞一段最短的

# 实际上"除 d=0 外全部待命"在约束体系下不可行（每个 UAV 必须出发并返回）
# 改为：d=0 跑全部，d=1,d=2 各跑最近的一个点后直接返回

print(f"【烂解 MIPStart】无人机 0 按顺序跑全部巡逻点 ({n_set} 条边)")
# d=1: 0 → 最近的点 → 0
# d=2: 0 → 次近的点 → 0

# 找离基地最近的两个巡逻点
dist_from_base = [(i, dist[0][i]) for i in range(1, n)]
dist_from_base.sort(key=lambda x: x[1])

for v in model3.getVars():
    if v.VarName.startswith('x['):
        parts = v.VarName.replace('x[', '').replace(']', '').split(',')
        d, i, j = int(parts[0]), int(parts[1]), int(parts[2])

        if d == 1:
            nearest = dist_from_base[0][0]  # 最近的点
            if (i == 0 and j == nearest) or (i == nearest and j == 0):
                v.Start = 1.0
                n_set += 1

        if d == 2:
            second = dist_from_base[1][0]  # 次近的点
            if (i == 0 and j == second) or (i == second and j == 0):
                v.Start = 1.0
                n_set += 1

print(f"  d=1: 基地 → P{dist_from_base[0][0]} → 基地")
print(f"  d=2: 基地 → P{dist_from_base[1][0]} → 基地")

t3, obj3, st3 = solve(model3, "烂解作 MIPStart")
model3.dispose()

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 55)
print(f"{'':<22} {'耗时':>10} {'目标值':>10}")
print("-" * 55)
for label, t, obj in [("冷启动", t1, obj1),
                       ("最优解 MIPStart", t2, obj2),
                       ("烂解 MIPStart", t3, obj3)]:
    obj_str = f"{obj:.2f}" if obj else "N/A"
    print(f"{label:<22} {t:>8.2f}s {obj_str:>10}")

print("-" * 55)
ratio_best = (t2 - t1) / t1 * 100 if t1 > 0 else 0
ratio_bad = (t3 - t1) / t1 * 100 if t1 > 0 else 0
print(f"最优解 vs 冷启动: {ratio_best:+.1f}%")
print(f"烂解   vs 冷启动: {ratio_bad:+.1f}%")
