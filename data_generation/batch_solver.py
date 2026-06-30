"""
批量求解器

支持多进程并行求解 MTSP 实例，带进度条和断点续传。
提取完整解信息用于 GNN 训练：变量取值、约束矩阵、模型结构。
"""
import os
import time
import pickle
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from model_builder import build_mtsp_model


def _solve_single_instance(args):
    """
    求解单个实例（模块级函数，供 multiprocessing.Pool 调用）

    Args:
        args: (instance, time_limit, output_dir, instance_idx, resume)

    Returns:
        tuple: (success: bool, status_msg: str)
    """
    instance, time_limit, output_dir, instance_idx, resume = args

    file_path = os.path.join(output_dir, f"data_{instance_idx:06d}.pkl")

    # 断点续传：跳过已存在的文件
    if resume and os.path.exists(file_path):
        return (True, "skipped")

    try:
        # 还原 numpy 数组
        base = np.array(instance['base'])
        patrol_points = np.array(instance['patrol_points'])
        num_uav = instance['num_uav']
        max_energy = instance['max_energy']

        # 构建模型
        model, metadata = build_mtsp_model(
            base, patrol_points, num_uav, max_energy
        )

        # 设置求解参数
        model.setParam('TimeLimit', time_limit)
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)  # 多进程下单线程更高效

        # 求解
        t_start = time.time()
        model.optimize()
        solve_time = time.time() - t_start

        # 判定解状态
        if model.Status == 2:  # GRB.OPTIMAL
            status = 'OPTIMAL'
            obj_val = model.ObjVal
        elif model.Status == 9:  # GRB.TIME_LIMIT
            if model.SolCount > 0:
                status = 'TIME_LIMIT'
                obj_val = model.ObjVal
            else:
                status = 'TIME_LIMIT_NO_SOL'
                obj_val = None
        elif model.Status == 3:  # GRB.INFEASIBLE
            status = 'INFEASIBLE'
            obj_val = None
        else:
            status = f'OTHER_{model.Status}'
            obj_val = model.ObjVal if model.SolCount > 0 else None

        # 提取 makespan
        makespan_val = None
        if model.SolCount > 0:
            t_var = model.getVarByName('makespan')
            if t_var is not None:
                makespan_val = float(t_var.X)

        # 提取 x 变量的二进制解
        x_values = {}
        if model.SolCount > 0:
            for v in model.getVars():
                if v.VarName.startswith('x['):
                    x_values[v.VarName] = int(round(v.X))
        else:
            # 无解时记录变量名，值设为 0
            for v in model.getVars():
                if v.VarName.startswith('x['):
                    x_values[v.VarName] = 0

        # 提取约束矩阵（必须在 model.update() 之后调用）
        model.update()
        A = model.getA()  # scipy.sparse.csr_matrix
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

        # 变量元信息
        all_vars = model.getVars()
        var_names = [v.VarName for v in all_vars]
        var_types = [v.VType for v in all_vars]
        var_lb = [float(v.LB) for v in all_vars]
        var_ub = [float(v.UB) for v in all_vars]

        # 约束统计
        num_linear_constrs = model.NumConstrs
        num_gen_constrs = model.NumGenConstrs

        # 组装解
        solution = {
            'status': status,
            'obj_val': obj_val,
            'makespan': makespan_val,
            'solve_time': solve_time,
            'x_values': x_values,
            'metadata': metadata,
            'model_info': {
                'num_vars': len(var_names),
                'num_linear_constrs': num_linear_constrs,
                'num_gen_constrs': num_gen_constrs,
                'num_int_vars': model.NumIntVars,
                'num_bin_vars': model.NumBinVars,
                'var_names': var_names,
                'var_types': var_types,
                'var_lb': var_lb,
                'var_ub': var_ub,
            },
            'constraint_matrix': constraint_matrix,
        }

        # 保存数据
        data = {'instance': instance, 'solution': solution}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        model.dispose()
        return (True, status)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        traceback_str = traceback.format_exc()

        # 保存错误信息
        try:
            error_data = {
                'instance': instance,
                'solution': {
                    'status': 'ERROR',
                    'error': error_msg,
                    'traceback': traceback_str,
                }
            }
            with open(file_path, 'wb') as f:
                pickle.dump(error_data, f)
        except Exception:
            pass

        return (False, error_msg)


def batch_solve(num_instances, output_dir, time_limit=60, num_workers=None,
                resume=True, instances=None):
    """
    批量生成并求解 MTSP 实例（多进程并行，带进度条和断点续传）

    工作流程：
      1. 生成随机实例（或使用传入的实例列表）
      2. 使用 multiprocessing.Pool 并行求解
      3. 保存每个实例为 .pkl 文件
      4. 输出统计摘要

    Args:
        num_instances: 实例总数
        output_dir: 输出目录路径
        time_limit: 每个实例求解时间上限（秒）
        num_workers: 并行进程数（None 则自动设为 min(cpu_count, 4)）
        resume: 是否跳过已存在的 .pkl 文件（断点续传）
        instances: 预生成的实例列表，为 None 时自动生成

    Returns:
        dict: 统计信息 {'total', 'optimal', 'time_limit', 'infeasible',
                       'error', 'skipped', 'other'}
    """
    from config import (BASE_SEED, MAP_SIZE, NUM_UAV_RANGE,
                        NUM_POINTS_RANGE, MAX_ENERGY_RANGE)
    from instance_generator import generate_instance_batch

    os.makedirs(output_dir, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), 4)

    # 生成实例
    if instances is None:
        instances = generate_instance_batch(
            num_instances, base_seed=BASE_SEED,
            map_size=MAP_SIZE, num_uav_range=NUM_UAV_RANGE,
            num_points_range=NUM_POINTS_RANGE, max_energy_range=MAX_ENERGY_RANGE
        )

    print(f"生成 {len(instances)} 个随机实例")
    print(f"使用 {num_workers} 个进程并行求解 (时间上限: {time_limit}s)")

    # 准备多进程参数
    args_list = [(inst, time_limit, output_dir, idx, resume)
                 for idx, inst in enumerate(instances)]

    # 多进程并行求解
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_solve_single_instance, args_list),
            total=len(args_list),
            desc="求解进度",
            unit="inst",
        ))

    # 汇总统计
    stats = {
        'total': len(instances),
        'optimal': 0,
        'time_limit': 0,
        'infeasible': 0,
        'error': 0,
        'skipped': 0,
        'other': 0,
    }

    for success, msg in results:
        if msg == "skipped":
            stats['skipped'] += 1
        elif not success:
            stats['error'] += 1
        elif msg == 'OPTIMAL':
            stats['optimal'] += 1
        elif msg.startswith('TIME_LIMIT'):
            stats['time_limit'] += 1
        elif msg == 'INFEASIBLE':
            stats['infeasible'] += 1
        else:
            stats['other'] += 1

    # 输出统计
    print(f"\n{'=' * 40}")
    print(f"总计: {stats['total']} 个实例")
    print(f"  OPTIMAL:            {stats['optimal']}")
    print(f"  TIME_LIMIT:         {stats['time_limit']}")
    print(f"  INFEASIBLE:         {stats['infeasible']}")
    print(f"  ERROR:              {stats['error']}")
    print(f"  其他状态:            {stats['other']}")
    if stats['skipped'] > 0:
        print(f"  (跳过已存在):        {stats['skipped']}")
    print(f"{'=' * 40}")

    return stats


def load_data(file_path):
    """
    加载并打印一个 .pkl 数据文件的关键信息

    Args:
        file_path: .pkl 文件路径

    Returns:
        dict: 加载的完整数据
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    instance = data['instance']
    solution = data['solution']

    print(f"实例: {instance['num_uav']} 架无人机, {instance['num_points']} 个巡逻点")
    print(f"状态: {solution.get('status')}")
    print(f"目标值: {solution.get('obj_val')}")
    print(f"Makespan: {solution.get('makespan')}")
    print(f"求解时间: {solution.get('solve_time', 0):.2f}s")

    if 'model_info' in solution:
        mi = solution['model_info']
        print(f"变量: {mi['num_vars']} (整数: {mi['num_int_vars']}, 二进制: {mi['num_bin_vars']})")
        print(f"线性约束: {mi['num_linear_constrs']},  通用约束: {mi['num_gen_constrs']}")

    if 'constraint_matrix' in solution and solution['constraint_matrix']:
        cm = solution['constraint_matrix']
        print(f"约束矩阵 A: {cm['shape'][0]} 行 x {cm['shape'][1]} 列, "
              f"{len(cm['data'])} 个非零元素")

    if 'x_values' in solution:
        xv = solution['x_values']
        ones = sum(1 for v in xv.values() if v == 1)
        print(f"x 变量: {len(xv)} 个, 其中 {ones} 个取值为 1")

    return data
