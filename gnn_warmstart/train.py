"""
训练脚本：二分类 BCE 损失，只对 x 变量计算
逐图训练，梯度累积，避免 DataLoader 兼容性问题
"""
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Subset

from config import *
from dataset import MTSPDataset, split_dataset
from gnn_model import WarmStartGNN


def compute_loss_and_metrics(logits, data, loss_fn):
    """计算损失和指标（只在标记为目标的变量上）"""
    mask = data.var_mask
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), 0.0, 0.0, 0.0

    targets = data.y[mask]
    outputs = logits[mask]

    loss = loss_fn(outputs, targets)

    with torch.no_grad():
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct = (preds == targets).float().sum()
        acc = correct.item() / mask.sum().item()

        tp = ((preds == 1) & (targets == 1)).float().sum().item()
        fp = ((preds == 1) & (targets == 0)).float().sum().item()
        fn = ((preds == 0) & (targets == 1)).float().sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

    return loss, acc, precision, recall


def train_one_epoch(model, indices, dataset, optimizer, loss_fn, device, grad_accum):
    """逐图训练一个 epoch，支持梯度累积"""
    model.train()
    total_loss, total_acc, total_prec, total_recall = 0.0, 0.0, 0.0, 0.0
    count = 0
    optimizer.zero_grad()

    random.shuffle(indices)

    for step, idx in enumerate(indices):
        graph = dataset[idx]
        if graph.num_vars == 0:
            continue
        graph = graph.to(device)

        logits = model(graph)
        loss, acc, prec, recall = compute_loss_and_metrics(logits, graph, loss_fn)
        loss = loss / grad_accum

        loss.backward()

        total_loss += loss.item() * grad_accum
        total_acc += acc
        total_prec += prec
        total_recall += recall
        count += 1

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

    if count % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    if count == 0:
        return 0, 0, 0, 0
    return total_loss / count, total_acc / count, total_prec / count, total_recall / count


@torch.no_grad()
def eval_set(model, indices, dataset, loss_fn, device):
    """逐图评估"""
    model.eval()
    total_loss, total_acc, total_prec, total_recall = 0.0, 0.0, 0.0, 0.0
    count = 0

    for idx in indices:
        graph = dataset[idx]
        if graph.num_vars == 0:
            continue
        graph = graph.to(device)

        logits = model(graph)
        loss, acc, prec, recall = compute_loss_and_metrics(logits, graph, loss_fn)

        total_loss += loss.item()
        total_acc += acc
        total_prec += prec
        total_recall += recall
        count += 1

    if count == 0:
        return 0, 0, 0, 0
    return total_loss / count, total_acc / count, total_prec / count, total_recall / count


def train(data_dir=None, output_dir=None, device=None):
    """完整训练流程"""
    if data_dir is None:
        data_dir = TRAIN_DATA_DIR
    if output_dir is None:
        output_dir = MODEL_SAVE_DIR
    if device is None:
        device = torch.device(DEVICE)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    print(f"加载数据集: {data_dir}")
    dataset = MTSPDataset(data_dir)
    print(f"  文件数: {len(dataset)}")

    # 统计有效图
    valid_count = 0
    for i in range(len(dataset)):
        if dataset[i].num_vars > 0:
            valid_count += 1
    print(f"  有效图: {valid_count} / {len(dataset)}")

    if valid_count == 0:
        print("错误: 没有有效图，请检查数据文件和 graph_builder")
        return None

    # 划分训练/验证/测试
    train_idx, val_idx, test_idx = split_dataset(
        dataset, TRAIN_RATIO, VAL_RATIO
    )
    print(f"  训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}")

    # 创建模型
    model = WarmStartGNN(
        var_feat_dim=7, constr_feat_dim=4,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_CONV_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_prec, train_recall = train_one_epoch(
            model, train_idx, dataset, optimizer, loss_fn, device, GRAD_ACCUM_STEPS
        )
        val_loss, val_acc, val_prec, val_recall = eval_set(
            model, val_idx, dataset, loss_fn, device
        )

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pt"))

    # 测试集评估
    print(f"\n最佳模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(
        os.path.join(output_dir, "best_model.pt"), weights_only=True
    ))
    test_loss, test_acc, test_prec, test_recall = eval_set(
        model, test_idx, dataset, loss_fn, device
    )
    print(f"测试集 | Loss: {test_loss:.4f} Acc: {test_acc:.4f} "
          f"Prec: {test_prec:.4f} Rec: {test_recall:.4f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_CONV_LAYERS,
            'dropout': DROPOUT,
        },
        'best_epoch': best_epoch,
        'test_metrics': {
            'loss': test_loss, 'acc': test_acc,
            'prec': test_prec, 'recall': test_recall,
        },
    }, os.path.join(output_dir, "model.pt"))

    print(f"模型已保存至: {output_dir}")
    return model
