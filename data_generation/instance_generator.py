"""
随机 MTSP 实例生成器
"""
import numpy as np


def generate_random_instance(seed, map_size=100.0, num_uav_range=(2, 4),
                              num_points_range=(5, 15), max_energy_range=(200, 400)):
    """
    生成一个随机的 MTSP 实例

    Args:
        seed: 随机种子
        map_size: 地图边长
        num_uav_range: 无人机数量范围 (min, max)
        num_points_range: 巡逻点数量范围 (min, max)
        max_energy_range: 最大能量范围 (min, max)

    Returns:
        dict: 包含所有输入参数的字典（均为 Python 原生类型或可序列化类型）
    """
    rng = np.random.RandomState(seed)

    num_uav = int(rng.randint(num_uav_range[0], num_uav_range[1] + 1))
    num_points = int(rng.randint(num_points_range[0], num_points_range[1] + 1))

    base = rng.uniform(0, map_size, size=2).tolist()
    patrol_points = rng.uniform(0, map_size, size=(num_points, 2)).tolist()
    max_energy = float(rng.uniform(max_energy_range[0], max_energy_range[1]))

    return {
        'seed': int(seed),
        'base': base,
        'patrol_points': patrol_points,
        'num_uav': num_uav,
        'max_energy': max_energy,
        'num_points': num_points,
        'map_size': map_size,
    }


def generate_instance_batch(num_instances, base_seed=42, **kwargs):
    """
    批量生成实例

    Args:
        num_instances: 实例数量
        base_seed: 起始随机种子
        **kwargs: 传递给 generate_random_instance 的其他参数

    Returns:
        list[dict]: 实例列表
    """
    instances = []
    for i in range(num_instances):
        inst = generate_random_instance(seed=base_seed + i, **kwargs)
        instances.append(inst)
    return instances
