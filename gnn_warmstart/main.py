"""
GNN 热启动入口：训练或评估

用法：
    python main.py --mode train                    # 训练模型
    python main.py --mode eval                     # 评估热启动效果
    python main.py --mode train --data ./data_dir  # 指定数据目录
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 父目录用于导入 data_generation 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import *
from train import train
from warmstart import evaluate_warmstart
from gnn_model import WarmStartGNN


def cmd_train(args):
    """训练 GNN 模型"""
    data_dir = args.data or TRAIN_DATA_DIR
    output_dir = args.output or MODEL_SAVE_DIR
    device = torch.device(args.device or DEVICE)

    print("=" * 60)
    print("训练 GNN 热启动模型")
    print(f"  数据目录: {data_dir}")
    print(f"  模型保存: {output_dir}")
    print(f"  设备: {device}")
    print(f"  隐藏维度: {HIDDEN_DIM}, 层数: {NUM_CONV_LAYERS}")
    print("=" * 60)

    model = train(data_dir=data_dir, output_dir=output_dir, device=device)
    print("\n训练完成！")
    return model


def cmd_eval(args):
    """评估热启动效果（需要已有训练数据中的最优解做标签对比）"""
    model_path = args.model or os.path.join(MODEL_SAVE_DIR, "best_model.pt")
    data_dir = args.data or TRAIN_DATA_DIR
    device = torch.device(args.device or DEVICE)

    # 加载模型
    print(f"加载模型: {model_path}")
    model = WarmStartGNN(
        var_feat_dim=7, constr_feat_dim=4,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_CONV_LAYERS,
        dropout=DROPOUT,
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("模型加载成功")

    # 从训练数据中选若干实例做评估
    from dataset import MTSPDataset
    dataset = MTSPDataset(data_dir)

    num_eval = min(args.num_eval, len(dataset))
    print(f"评估 {num_eval} 个实例...")

    results = {'warm_optimal': 0, 'cold_optimal': 0,
               'warm_obj': [], 'cold_obj': [],
               'warm_time': [], 'cold_time': []}

    for i in range(num_eval):
        graph = dataset[i]
        if graph.num_vars == 0:
            continue

        # 需要原始的 instance 字典
        import pickle
        fname = dataset._get_file_list()[i]
        with open(os.path.join(data_dir, fname), 'rb') as f:
            raw = pickle.load(f)
        instance = raw['instance']

        result = evaluate_warmstart(
            model, instance, time_limit=args.time_limit, device='device'
        )

        warm_res = result['warm']
        cold_res = result['cold']

        print(f"\n实例 {i}: "
              f"暖启动 {warm_res['status']} obj={warm_res['obj_val']} "
              f"{warm_res['solve_time']:.1f}s | "
              f"冷启动 {cold_res['status']} obj={cold_res['obj_val']} "
              f"{cold_res['solve_time']:.1f}s")

        if warm_res['status'] == 'OPTIMAL':
            results['warm_optimal'] += 1
        if cold_res['status'] == 'OPTIMAL':
            results['cold_optimal'] += 1

        if warm_res['obj_val'] is not None:
            results['warm_obj'].append(warm_res['obj_val'])
            results['warm_time'].append(warm_res['solve_time'])
        if cold_res['obj_val'] is not None:
            results['cold_obj'].append(cold_res['obj_val'])
            results['cold_time'].append(cold_res['solve_time'])

    # 汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print(f"  暖启动 OPTIMAL: {results['warm_optimal']}/{num_eval}")
    print(f"  冷启动 OPTIMAL: {results['cold_optimal']}/{num_eval}")
    if results['warm_obj']:
        print(f"  暖启动平均目标值: {sum(results['warm_obj'])/len(results['warm_obj']):.2f}")
        print(f"  暖启动平均时间: {sum(results['warm_time'])/len(results['warm_time']):.2f}s")
    if results['cold_obj']:
        print(f"  冷启动平均目标值: {sum(results['cold_obj'])/len(results['cold_obj']):.2f}")
        print(f"  冷启动平均时间: {sum(results['cold_time'])/len(results['cold_time']):.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN 热启动训练 & 评估")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='模式: train (训练) 或 eval (评估)')
    parser.add_argument('--data', type=str, default=None,
                        help='训练/评估数据目录')
    parser.add_argument('--output', type=str, default=None,
                        help='模型保存目录（仅训练模式）')
    parser.add_argument('--model', type=str, default=None,
                        help='模型文件路径（仅评估模式）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cpu/cuda)')
    parser.add_argument('--num_eval', type=int, default=5,
                        help='评估实例数量（仅评估模式）')
    parser.add_argument('--time_limit', type=int, default=60,
                        help='求解时间上限（仅评估模式）')
    args = parser.parse_args()

    if args.mode == 'train':
        cmd_train(args)
    elif args.mode == 'eval':
        cmd_eval(args)
