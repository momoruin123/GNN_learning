"""
MTSP 数据生成配置文件
"""

# ========== 输出配置 ==========
OUTPUT_DIR = "./training_data"  # 数据保存目录
NUM_INSTANCES = 10              # 生成实例总数
TIME_LIMIT = 120                # 每个实例的求解时间上限（秒）
NUM_WORKERS = 4                 # 并行求解进程数

# ========== 实例生成参数 ==========
MAP_SIZE = 100.0                # 地图边长
NUM_UAV_RANGE = (3, 6)          # 无人机数量范围 [min, max]
NUM_POINTS_RANGE = (15, 30)     # 巡逻点数量范围 [min, max]
MAX_ENERGY_RANGE = (200, 400)   # 最大能量范围 [min, max]
ENERGY_RATE = 1.0               # 单位距离能量消耗

# ========== 随机种子 ==========
BASE_SEED = 42                  # 起始随机种子

# ========== 断点续传 ==========
RESUME = True                   # 是否跳过已存在的文件
