import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle
import os
import time
from multiprocessing import Pool, cpu_count
import build_mtsp_model

def generate_random_instance(seed, num_uav_range=(2, 4), num_points_range=(5, 12), map_size=100.0, max_energy_range=(200, 400)):
    """
    生成一个随机的 MTSP 实例
    """
    np.random.seed(seed)
    
    num_uav = np.random.randint(num_uav_range[0], num_uav_range[1] + 1)
    num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
    
    base = np.random.uniform(0, map_size, size=2)
    patrol_points = np.random.uniform(0, map_size, size=(num_points, 2))
    max_energy = np.random.uniform(max_energy_range[0], max_energy_range[1])
    
    return {
        'seed': seed,
        'base': base,
        'patrol_points': patrol_points,
        'num_uav': num_uav,
        'max_energy': max_energy,
        'num_points': num_points,
        'map_size': map_size
    }


def solve_instance(instance_data, time_limit=60, verbose=False):
    """
    求解单个实例
    """
    base = instance_data['base']
    patrol_points = instance_data['patrol_points']
    num_uav = instance_data['num_uav']
    max_energy = instance_data['max_energy']
    
    try:
        model, model_data = build_mtsp_model(base, patrol_points, num_uav, max_energy)
        model.setParam('TimeLimit', time_limit)
        model.setParam('OutputFlag', 1 if verbose else 0)
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        # 提取解
        if model.Status == GRB.OPTIMAL:
            status = 'OPTIMAL'
        elif model.Status == GRB.TIME_LIMIT:
            status = 'TIME_LIMIT'
        else:
            status = 'OTHER'
        
        # 提取变量取值（转换为普通 Python 类型，方便 pickle）
        x_values = {}
        for d in range(num_uav):
            for i in range(model_data['n']):
                for j in range(model_data['n']):
                    if i != j:
                        x_values[f'x_{d}_{i}_{j}'] = float(model_data['vars'][d, i, j].X) if hasattr(model_data['vars'][d, i, j], 'X') else 0.0
        
        # 提取所有非零变量
        binary_solution = {}
        for key, val in x_values.items():
            if val > 0.5:
                binary_solution[key] = 1
            else:
                binary_solution[key] = 0
        
        solution = {
            'status': status,
            'obj_val': model.ObjVal if model.ObjVal is not None else None,
            'solve_time': solve_time,
            'makespan': t.X if hasattr(t, 'X') else None,
            'x_values': binary_solution,  # 保存二进制解
            'model_data': {
                'n': model_data['n'],
                'dist': model_data['dist'].tolist(),  # 转为 list 方便 pickle
                'points': model_data['points'].tolist(),
            }
        }
        
        return solution
    
    except Exception as e:
        print(f"求解失败: {e}")
        return {
            'status': 'ERROR',
            'error': str(e),
            'solve_time': 0
        }


def generate_batch(num_instances, output_dir, time_limit=60, num_workers=None):
    """
    批量生成数据（支持多进程）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # 限制最多 4 个进程，避免内存爆炸
    
    # 生成实例
    print(f"🔍 生成 {num_instances} 个随机实例...")
    instances = []
    for i in range(num_instances):
        inst = generate_random_instance(i)
        instances.append(inst)
    
    # 批量求解
    print(f"🚀 使用 {num_workers} 个进程求解...")
    
    # 准备参数
    args = [(inst, time_limit) for inst in instances]
    
    with Pool(num_workers) as pool:
        results = pool.starmap(solve_instance, args)
    
    # 保存结果
    print(f"💾 保存结果到 {output_dir}")
    success_count = 0
    for i, (instance, solution) in enumerate(zip(instances, results)):
        data = {
            'instance': instance,
            'solution': solution
        }
        file_path = os.path.join(output_dir, f"data_{i:06d}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        if solution.get('status') == 'OPTIMAL':
            success_count += 1
    
    print(f"✅ 完成！成功求解 {success_count}/{num_instances} 个实例")
    return results


def check_data(file_path):
    """查看数据内容"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    instance = data['instance']
    solution = data['solution']
    print(f"实例: {instance['num_uav']} 架无人机, {instance['num_points']} 个巡逻点")
    print(f"状态: {solution.get('status')}")
    print(f"目标值: {solution.get('obj_val')}")
    print(f"求解时间: {solution.get('solve_time')}")
    return data


# ---------- 主程序 ----------
if __name__ == "__main__":
    # 参数配置
    OUTPUT_DIR = "./training_data"
    NUM_INSTANCES = 100  # 先测试 100 个
    TIME_LIMIT = 60
    
    # 生成数据
    generate_batch(NUM_INSTANCES, OUTPUT_DIR, TIME_LIMIT)
    
    # 检查一个数据
    print("\n📊 检查数据样本:")
    check_data(os.path.join(OUTPUT_DIR, "data_000000.pkl"))