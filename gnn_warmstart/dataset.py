"""
数据集：从 .pkl 文件加载 MILP 数据并转换为 PyG Data 对象
"""
import os
import pickle
import torch
from torch_geometric.data import Dataset

from graph_builder import milp_to_graph


class MTSPDataset(Dataset):
    """
    MTSP MILP 图数据集

    每个样本是一个 .pkl 文件，包含 instance（输入参数）和 solution（求解结果）。
    通过 milp_to_graph 将 solution 中的 MILP 结构转换为二分图。

    Args:
        data_dir: .pkl 文件所在目录
        file_list: 指定要加载的文件名列表（None 则加载目录中所有 .pkl）
        transform: PyG 变换（可选）
    """

    def __init__(self, data_dir, file_list=None, transform=None):
        self.data_dir = data_dir
        self._file_list = file_list
        self._cached_files = None  # 缓存文件列表
        super().__init__(root=None, transform=transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def _get_file_list(self):
        if self._cached_files is None:
            if self._file_list is not None:
                self._cached_files = list(self._file_list)
            else:
                self._cached_files = sorted(
                    [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
                )
        return self._cached_files

    def len(self):
        return len(self._get_file_list())

    def get(self, idx):
        files = self._get_file_list()
        fname = files[idx]

        file_path = os.path.join(self.data_dir, fname)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        solution = data['solution']

        # 跳过求解失败的实例
        if solution.get('status') in ('ERROR',):
            # 返回一个占位图
            graph = type('DummyGraph', (), {})()
            graph.num_nodes = 0
            graph.num_vars = 0
            graph.x = torch.zeros(0, 7)
            graph.edge_index = torch.zeros(2, 0, dtype=torch.long)
            graph.edge_attr = torch.zeros(0, 1)
            graph.y = torch.zeros(0)
            graph.var_mask = torch.zeros(0, dtype=torch.bool)
            graph.var_names = []
            return graph

        try:
            graph = milp_to_graph(solution)
        except Exception as e:
            import traceback
            print(f"[WARNING] 构建图失败 {fname}: {e}")
            traceback.print_exc()
            graph = type('DummyGraph', (), {})()
            graph.num_nodes = 0
            graph.num_vars = 0
            graph.x = torch.zeros(0, 7)
            graph.edge_index = torch.zeros(2, 0, dtype=torch.long)
            graph.edge_attr = torch.zeros(0, 1)
            graph.y = torch.zeros(0)
            graph.var_mask = torch.zeros(0, dtype=torch.bool)
            graph.var_names = []
        return graph


def collate_filter_empty(batch):
    """过滤掉空图的 batch 整理函数"""
    valid = [d for d in batch if d.num_vars > 0]
    if len(valid) == 0:
        return None
    from torch_geometric.data import Batch
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
