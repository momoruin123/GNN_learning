"""
数据集：从 .pkl 文件加载 MILP 数据并转换为 PyG Data 对象
"""
import os
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from graph_builder import milp_to_graph


def _empty_graph():
    """构造一个有效的空 Data 对象，训练时会被 collate_fn 过滤"""
    g = Data()
    g.num_nodes = 0
    g.num_vars = 0
    g.x = torch.zeros(0, 7)
    g.edge_index = torch.zeros(2, 0, dtype=torch.long)
    g.edge_attr = torch.zeros(0, 1)
    g.y = torch.zeros(0)
    g.var_mask = torch.zeros(0, dtype=torch.bool)
    g.var_names = []
    return g


class MTSPDataset(Dataset):
    """
    MTSP MILP 图数据集

    每个样本是一个 .pkl 文件，包含 instance（输入参数）和 solution（求解结果）。
    通过 milp_to_graph 将 solution 中的 MILP 结构转换为二分图。

    Args:
        data_dir: .pkl 文件所在目录
        file_list: 指定要加载的文件名列表（None 则加载目录中所有 .pkl）
    """

    def __init__(self, data_dir, file_list=None):
        self.data_dir = data_dir
        self._file_list = file_list
        self._cached_files = None

    def _get_file_list(self):
        if self._cached_files is None:
            if self._file_list is not None:
                self._cached_files = list(self._file_list)
            else:
                self._cached_files = sorted(
                    [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
                )
        return self._cached_files

    def __len__(self):
        return len(self._get_file_list())

    def __getitem__(self, idx):
        files = self._get_file_list()
        fname = files[idx]

        file_path = os.path.join(self.data_dir, fname)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        solution = data.get('solution', {})
        status = solution.get('status', 'UNKNOWN')

        if status == 'ERROR':
            err_detail = solution.get('error', 'no detail')
            print(f"[SKIP] {fname}: status=ERROR, reason={err_detail}")
            return _empty_graph()

        if status in ('TIME_LIMIT_NO_SOL', 'INFEASIBLE'):
            print(f"[SKIP] {fname}: status={status}, 无可用解")
            return _empty_graph()

        mi = solution.get('model_info', {})
        if not mi or mi.get('num_vars', 0) == 0:
            print(f"[SKIP] {fname}: model_info 缺失或 num_vars=0, keys={list(mi.keys())[:5]}")
            return _empty_graph()

        try:
            graph = milp_to_graph(solution)
        except Exception as e:
            import traceback
            print(f"[FAIL] {fname}: {e}")
            traceback.print_exc()
            return _empty_graph()

        return graph


def collate_filter_empty(batch):
    """过滤掉空图的 batch 整理函数"""
    valid = [d for d in batch if d.num_vars > 0]
    if len(valid) == 0:
        return None
    return Batch.from_data_list(valid)


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """将数据集按比例划分为训练/验证/测试集"""
    num = len(dataset)
    indices = list(range(num))
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num, generator=rng).tolist()

    train_end = int(num * train_ratio)
    val_end = int(num * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx
