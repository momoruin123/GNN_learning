"""
热启动：使用训练好的 GNN 为新的 MTSP 实例生成 MIP 初始解

工作流程：
  1. 构建 MILP 模型（不求解）
  2. 提取 MILP 结构（约束矩阵、变量信息等）
  3. 转换为二分图
  4. GNN 推理 → 预测每个 x 变量的值
  5. 将预测值作为 MIPStart 设置到 Gurobi 模型
  6. 求解（有/无热启动对比）
"""
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def extract_milp_structure(model):
    """
    从不求解的模型中提取结构信息（与 batch_solver 提取的格式一致）

    Args:
        model: gurobipy.Model（已构建，可未求解）
    Returns:
        dict: 包含 model_info、constraint_matrix 等字段
    """
    model.update()
    A = model.getA()

    if A is not None and A.nnz > 0:
        A_coo = A.tocoo()
        constraint_matrix = {
            'row': A_coo.row.tolist(),
            'col': A_coo.col.tolist(),
            'data': A_coo.data.tolist(),
            'shape': list(A_coo.shape),
        }
    else:
        constraint_matrix = None

    all_vars = model.getVars()
    var_names = [v.VarName for v in all_vars]
    var_types = [v.VType for v in all_vars]
    var_lb = [float(v.LB) for v in all_vars]
    var_ub = [float(v.UB) for v in all_vars]
    var_obj = [float(v.Obj) for v in all_vars]

    all_constrs = model.getConstrs()
    constr_rhs = [float(c.RHS) for c in all_constrs]
    constr_sense = [c.Sense for c in all_constrs]

    return {
        'model_info': {
            'num_vars': len(var_names),
            'num_linear_constrs': model.NumConstrs,
            'num_gen_constrs': model.NumGenConstrs,
            'num_int_vars': model.NumIntVars,
            'num_bin_vars': model.NumBinVars,
            'var_names': var_names,
            'var_types': var_types,
            'var_lb': var_lb,
            'var_ub': var_ub,
            'var_obj': var_obj,
            'constr_rhs': constr_rhs,
            'constr_sense': constr_sense,
        },
        'constraint_matrix': constraint_matrix,
        'x_values': {},  # 推理时无标签
    }


def set_mip_start_from_predictions(model, var_names, probs, num_patrol, num_uav):
    """
    将 GNN 预测概率转为 MIPStart：取概率最高的 top-K 个 x 变量设为 1

    K = num_patrol + 2*num_uav  （巡逻点入边 + 每架 UAV 的出发/返回边）
    """
    all_vars = model.getVars()
    var_dict = {v.VarName: v for v in all_vars}

    # 只收集 x 变量的概率
    x_probs = []  # (prob, var_name)
    for name, prob in zip(var_names, probs):
        if name.startswith('x['):
            x_probs.append((prob, name))

    # 降序排列，取 top K
    x_probs.sort(key=lambda v: v[0], reverse=True)
    K = min(num_patrol + 2 * num_uav, len(x_probs))

    num_set = 0
    for prob, name in x_probs[:K]:
        if name in var_dict:
            var_dict[name].Start = 1.0
            num_set += 1

    print(f"  MIPStart 设置: {num_set} 个变量 = 1 (top {K} / {len(x_probs)} x 变量)")



def solve_with_warmstart(model, var_names, probs, num_patrol, num_uav,
                         time_limit=60, verbose=False):
    """
    使用 GNN 预测概率作为热启动求解（top-K 策略）
    """
    set_mip_start_from_predictions(model, var_names, probs, num_patrol, num_uav)

    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1 if verbose else 0)

    t_start = time.time()
    model.optimize()
    solve_time = time.time() - t_start

    if model.Status == GRB.OPTIMAL:
        status = 'OPTIMAL'
        obj_val = model.ObjVal
    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            status = 'TIME_LIMIT'
            obj_val = model.ObjVal
        else:
            status = 'TIME_LIMIT_NO_SOL'
            obj_val = None
    else:
        status = f'OTHER_{model.Status}'
        obj_val = model.ObjVal if model.SolCount > 0 else None

    return {
        'status': status,
        'obj_val': obj_val,
        'solve_time': solve_time,
    }


def solve_cold(model, time_limit=60, verbose=False):
    """
    无热启动求解（基线对照）
    """
    # 清除任何已有的 MIPStart
    for v in model.getVars():
        v.Start = GRB.UNDEFINED

    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1 if verbose else 0)

    t_start = time.time()
    model.optimize()
    solve_time = time.time() - t_start

    if model.Status == GRB.OPTIMAL:
        status = 'OPTIMAL'
        obj_val = model.ObjVal
    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            status = 'TIME_LIMIT'
            obj_val = model.ObjVal
        else:
            status = 'TIME_LIMIT_NO_SOL'
            obj_val = None
    else:
        status = f'OTHER_{model.Status}'
        obj_val = model.ObjVal if model.SolCount > 0 else None

    return {
        'status': status,
        'obj_val': obj_val,
        'solve_time': solve_time,
    }


def evaluate_warmstart(gnn_model, instance, time_limit=60, device='cpu'):
    """
    对单个实例评估 GNN 热启动效果

    Args:
        gnn_model: 训练好的 WarmStartGNN
        instance: 实例字典 {'base', 'patrol_points', 'num_uav', 'max_energy'}
        time_limit: 求解时间上限
        device: 推理设备
    Returns:
        dict: {warm_result, cold_result, prediction_stats}
    """
    import torch
    from data_generation.model_builder import build_mtsp_model
    from graph_builder import milp_to_graph

    base = np.array(instance['base'])
    patrol_points = np.array(instance['patrol_points'])
    num_uav = instance['num_uav']
    max_energy = instance['max_energy']

    # 构建模型并提取结构（先求解获取最优标签，用于评估预测精度）
    # 但实际上我们做推理时不应该先求解...

    # 构建一份模型用于推理 + 热启动
    model_ws, _ = build_mtsp_model(base, patrol_points, num_uav, max_energy)
    milp_struct = extract_milp_structure(model_ws)

    # 构建图并推理
    graph = milp_to_graph(milp_struct)
    graph = graph.to(device)

    gnn_model.eval()
    gnn_model.to(device)
    with torch.no_grad():
        logits = gnn_model(graph)
        probs = torch.sigmoid(logits)

    # 热启动求解（top-K 策略）
    var_names = milp_struct['model_info']['var_names']
    warm_result = solve_with_warmstart(
        model_ws, var_names, probs.cpu().numpy(),
        num_patrol=instance['num_points'], num_uav=num_uav,
        time_limit=time_limit
    )
    model_ws.dispose()

    # 冷启动求解（无热启动）
    model_cold, _ = build_mtsp_model(base, patrol_points, num_uav, max_energy)
    cold_result = solve_cold(model_cold, time_limit)
    model_cold.dispose()

    # 统计预测
    K = instance['num_points'] + 2 * num_uav
    num_ones = K

    return {
        'warm': warm_result,
        'cold': cold_result,
        'pred_stats': {
            'total_vars': len(probs),
            'predicted_ones': num_ones,
        },
    }
