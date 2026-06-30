"""
MTSP MILP 模型构建器

封装现有的 MTSP 模型为可调用函数，返回 Gurobi 模型对象和可序列化的元数据。
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_mtsp_model(base, patrol_points, num_uav, max_energy=300, energy_rate=1.0):
    """
    构建 MTSP MILP 模型

    模型描述：
      - 变量: x[d][i][j] (二进制，无人机 d 从 i 飞到 j)
              u[d][i] (连续，MTZ 顺序号)
              t (连续，makespan)
              e[d][i] (连续，剩余能量)
      - 约束: 流量平衡、巡逻点单次访问、基地出发/返回、
              MTZ 子回路消除、能量消耗 (Indicator约束)、Makespan 定义
      - 目标: 最小化 Makespan

    Args:
        base: 基地坐标，shape (2,) 或 list
        patrol_points: 巡逻点坐标，shape (num_points, 2) 或 list
        num_uav: 无人机数量
        max_energy: 每架无人机最大能量
        energy_rate: 单位距离能量消耗

    Returns:
        model: gurobipy.Model 对象
        metadata: dict，包含模型结构信息（不含 Gurobi 对象，可安全序列化）
            - n: 节点总数（基地 + 巡逻点）
            - num_uav: 无人机数量
            - max_energy: 最大能量
            - energy_rate: 能量消耗率
            - dist: 距离矩阵 (list of list)
            - points: 所有点坐标 (list of list)
    """
    points = np.vstack([np.array(base).reshape(1, -1),
                        np.array(patrol_points).reshape(-1, 2)])
    n = len(points)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(points[i] - points[j])

    model = gp.Model("MTSP")

    # ---------- 变量 ----------
    x = model.addVars(num_uav, n, n, vtype=GRB.BINARY, name='x')
    u = model.addVars(num_uav, n, vtype=GRB.CONTINUOUS, lb=0, ub=n - 1, name='u')
    t = model.addVar(lb=0, name="makespan")
    e = model.addVars(num_uav, n, vtype=GRB.CONTINUOUS, lb=0, ub=max_energy, name='e')

    # ---------- 约束 ----------
    # 1. 流量平衡: 进入节点 i 的流量 = 离开节点 i 的流量
    for d in range(num_uav):
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[d, i, j] for j in range(n)) ==
                gp.quicksum(x[d, j, i] for j in range(n)),
                name=f"flow_balance_{d}_{i}"
            )

    # 2. 每个巡逻点被恰好访问一次
    for i in range(1, n):
        model.addConstr(
            gp.quicksum(x[d, i, j] for d in range(num_uav) for j in range(n) if j != i) == 1,
            name=f"visited_once_{i}"
        )

    # 3. 禁止自环
    for d in range(num_uav):
        for i in range(n):
            model.addConstr(x[d, i, i] == 0, name=f"no_self_loop_{d}_{i}")

    # 4. 每架无人机从基地出发一次
    for d in range(num_uav):
        model.addConstr(
            gp.quicksum(x[d, 0, j] for j in range(1, n)) == 1,
            name=f"start_{d}"
        )

    # 5. 每架无人机返回基地一次
    for d in range(num_uav):
        model.addConstr(
            gp.quicksum(x[d, i, 0] for i in range(1, n)) == 1,
            name=f"end_{d}"
        )

    # 6. MTZ 子回路消除
    for d in range(num_uav):
        model.addConstr(u[d, 0] == 0, name=f"mtz_base_{d}")
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(
                        u[d, i] - u[d, j] + n * x[d, i, j] <= n - 1,
                        name=f"mtz_{d}_{i}_{j}"
                    )

    # 7. Makespan 约束: 每架无人机的总路径长度 <= t
    for d in range(num_uav):
        model.addConstr(
            gp.quicksum(dist[i, j] * x[d, i, j]
                        for i in range(n) for j in range(n) if i != j) <= t,
            name=f"makespan_{d}"
        )

    # 8. 基地初始能量
    for d in range(num_uav):
        model.addConstr(e[d, 0] == max_energy, name=f"base_energy_{d}")

    # 9. 能量消耗约束 (Indicator 约束)
    for d in range(num_uav):
        for i in range(n):
            for j in range(n):
                if i != j:
                    if j != 0:
                        # 飞往巡逻点: e[d,j] = e[d,i] - dist[i,j] * energy_rate
                        model.addGenConstrIndicator(
                            x[d, i, j], True,
                            e[d, j] == e[d, i] - dist[i][j] * energy_rate,
                            name=f"energy_to_point_{d}_{i}_{j}"
                        )
                    else:
                        # 返回基地: e[d,i] - dist[i,0]*energy_rate >= 0
                        model.addGenConstrIndicator(
                            x[d, i, 0], True,
                            e[d, i] - dist[i][0] * energy_rate >= 0,
                            name=f"energy_to_base_{d}_{i}"
                        )

    # ---------- 目标 ----------
    model.setObjective(t, GRB.MINIMIZE)

    # ---------- 元数据（不含 Gurobi 对象，可安全 pickle）----------
    metadata = {
        'n': int(n),
        'num_uav': int(num_uav),
        'max_energy': float(max_energy),
        'energy_rate': float(energy_rate),
        'dist': dist.tolist(),
        'points': points.tolist(),
    }

    return model, metadata
