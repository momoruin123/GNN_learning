"""
MTSP 数据生成入口脚本

用于批量生成 GNN 热启动训练数据：
  1. 随机生成 MTSP 实例
  2. 使用 Gurobi 并行求解
  3. 保存实例和求解结果到 .pkl 文件

使用方式：
    python main.py                     # 使用 config.py 中的默认配置
    python main.py --instances 100     # 命令行覆盖实例数量
"""
import os
import sys
import argparse

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from batch_solver import batch_solve, load_data


if __name__ == "__main__":
    # 命令行参数（覆盖 config.py 中的值）
    parser = argparse.ArgumentParser(description="MTSP GNN 训练数据批量生成")
    parser.add_argument('--instances', type=int, default=NUM_INSTANCES,
                        help=f'实例数量 (默认: {NUM_INSTANCES})')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    parser.add_argument('--time_limit', type=int, default=TIME_LIMIT,
                        help=f'求解时间上限 (默认: {TIME_LIMIT}s)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'并行进程数 (默认: {NUM_WORKERS})')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用断点续传，重新求解所有实例')
    args = parser.parse_args()

    # 配置汇总
    resume = RESUME if not args.no_resume else False

    print("=" * 60)
    print("MTSP GNN 训练数据批量生成")
    print("=" * 60)
    print(f"配置:")
    print(f"  输出目录:        {args.output}")
    print(f"  实例数量:        {args.instances}")
    print(f"  求解时间上限:     {args.time_limit}s")
    print(f"  并行进程数:       {args.workers}")
    print(f"  无人机范围:       {NUM_UAV_RANGE}")
    print(f"  巡逻点范围:       {NUM_POINTS_RANGE}")
    print(f"  最大能量范围:     {MAX_ENERGY_RANGE}")
    print(f"  地图大小:         {MAP_SIZE}")
    print(f"  断点续传:         {'是' if resume else '否'}")
    print("=" * 60)

    # 批量求解
    stats = batch_solve(
        num_instances=args.instances,
        output_dir=args.output,
        time_limit=args.time_limit,
        num_workers=args.workers,
        resume=resume,
    )

    # 检查第一个样本
    print("\n" + "=" * 60)
    print("数据样本检查 (第一个):")
    sample_file = os.path.join(args.output, "data_000000.pkl")
    if os.path.exists(sample_file):
        load_data(sample_file)
    else:
        print("  (无可用样本)")
