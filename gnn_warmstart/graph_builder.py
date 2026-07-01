"""
图构建器：将 MILP 实例转换为 PyG 异构图（变量-约束二分图）

变量节点（0 到 V-1）：
  特征: [var_type_onehot(3), lb_norm, ub_norm, obj_coeff_norm, is_target(1)]

约束节点（V 到 V+C-1）：
  特征: [sense_onehot(3), rhs_norm]

边（双向）：
  特征: [coeff_norm]
"""
import torch
import numpy as np
from torch_geometric.data import Data


# var_type: Gurobi vtype char → one-hot index
_VTYPE_MAP = {'C': 0, 'I': 1, 'B': 2}  # CONTINUOUS, INTEGER, BINARY
# constr_sense: Gurobi sense char → one-hot index
_SENSE_MAP = {'<': 0, '>': 1, '=': 2}  # LE, GE, EQ


def _safe_normalize(values, scale=None):
    """归一化到 [-1, 1] 或按给定 scale 缩放，避免除零"""
    arr = np.asarray(values, dtype=np.float32)
    if scale is None:
        scale = np.max(np.abs(arr))
    if scale < 1e-8:
        return arr, 1.0
    return arr / scale, scale


def _one_hot(values, mapping, num_classes):
    """手动 one-hot 编码"""
    n = len(values)
    mat = np.zeros((n, num_classes), dtype=np.float32)
    for i, v in enumerate(values):
        idx = mapping.get(v, 0)
        mat[i, idx] = 1.0
    return mat


def milp_to_graph(solution_data, normalize=True):
    """
    将求解结果中的 MILP 结构转换为 PyG Data 对象

    Data 对象字段：
      - x:           [num_nodes, feat_dim] 节点特征
      - edge_index:  [2, num_edges] 边的源/目标索引
      - edge_attr:   [num_edges, 1] 边特征（系数值）
      - y:           [num_vars] 目标标签（仅对 x 变量有效，其余为 0）
      - var_mask:    [num_vars] 布尔掩码，标记需要预测二值的变量
      - num_vars:    int 变量节点数
      - num_constrs: int 约束节点数
      - var_names:   list[str] 变量名列表
      - instance_id: str 实例标识

    Args:
        solution_data: dict，即 .pkl 文件加载后的 data['solution']
        normalize: 是否归一化特征

    Returns:
        torch_geometric.data.Data
    """
    mi = solution_data['model_info']
    cm = solution_data['constraint_matrix']
    x_values = solution_data['x_values']
    metadata = solution_data.get('metadata', {})

    V = mi['num_vars']
    C = mi['num_linear_constrs']

    # 1. 构建变量节点特征
    var_types = mi['var_types']  # ['C', 'B', ...]
    var_lb = np.array(mi['var_lb'], dtype=np.float32)
    var_ub = np.array(mi['var_ub'], dtype=np.float32)
    var_obj = np.array(mi['var_obj'], dtype=np.float32)

    type_onehot = _one_hot(var_types, _VTYPE_MAP, 3)

    # 归一化
    if normalize and V > 0:
        lb_norm, _ = _safe_normalize(var_lb)
        ub_norm, _ = _safe_normalize(var_ub)
        obj_norm, _ = _safe_normalize(var_obj)
    else:
        lb_norm, ub_norm, obj_norm = var_lb, var_ub, var_obj

    # is_target: 标记哪些变量是 x 变量（需要预测的二进制路由变量）
    var_names = mi['var_names']
    is_target = np.array([n.startswith('x[') for n in var_names], dtype=np.float32)

    var_feat_list = [type_onehot,
                     lb_norm.reshape(-1, 1),
                     ub_norm.reshape(-1, 1),
                     obj_norm.reshape(-1, 1),
                     is_target.reshape(-1, 1)]
    var_feat = np.concatenate(var_feat_list, axis=1)  # [V, 7]

    # 2. 构建约束节点特征
    constr_sense = mi['constr_sense']  # ['<', '=', ...]
    constr_rhs = np.array(mi['constr_rhs'], dtype=np.float32)

    sense_onehot = _one_hot(constr_sense, _SENSE_MAP, 3)

    if normalize and C > 0:
        rhs_norm, _ = _safe_normalize(constr_rhs)
    else:
        rhs_norm = constr_rhs

    constr_feat = np.concatenate(
        [sense_onehot, rhs_norm.reshape(-1, 1)], axis=1)  # [C, 4]

    # 3. 合并节点特征
    # 大图: 节点 0..V-1 是变量, V..V+C-1 是约束
    total_nodes = V + C
    feat_dim = var_feat.shape[1]  # 7

    if C > 0:
        assert constr_feat.shape[1] <= feat_dim, \
            f"约束特征维度 {constr_feat.shape[1]} > 变量特征维度 {feat_dim}"
        # 补齐到相同维度
        constr_feat_padded = np.zeros((C, feat_dim), dtype=np.float32)
        constr_feat_padded[:, :constr_feat.shape[1]] = constr_feat
    else:
        constr_feat_padded = np.zeros((0, feat_dim), dtype=np.float32)

    x = np.vstack([var_feat, constr_feat_padded])  # [V+C, feat_dim]

    # 4. 构建边（双向：V→C 和 C→V）
    if cm is not None and len(cm['row']) > 0:
        row_constraints = np.array(cm['row'], dtype=np.int64)  # 约束索引 0..C-1
        col_variables = np.array(cm['col'], dtype=np.int64)    # 变量索引 0..V-1
        edge_data_raw = np.array(cm['data'], dtype=np.float32)
    else:
        row_constraints = np.array([], dtype=np.int64)
        col_variables = np.array([], dtype=np.int64)
        edge_data_raw = np.array([], dtype=np.float32)

    num_edges_raw = len(edge_data_raw)

    # 正向: 变量 v → 约束 c+V
    edge_src_forward = col_variables
    edge_dst_forward = row_constraints + V

    # 反向: 约束 c+V → 变量 v
    edge_src_backward = row_constraints + V
    edge_dst_backward = col_variables

    edge_index = np.stack([
        np.concatenate([edge_src_forward, edge_src_backward]),
        np.concatenate([edge_dst_forward, edge_dst_backward]),
    ], axis=0)  # [2, 2*num_edges]

    # 边特征（双向用相同系数值）
    if normalize and num_edges_raw > 0:
        edge_feat_norm, _ = _safe_normalize(edge_data_raw)
    else:
        edge_feat_norm = edge_data_raw

    edge_attr = np.concatenate([edge_feat_norm, edge_feat_norm]).reshape(-1, 1)

    # 5. 构建标签 y
    y = np.zeros(V, dtype=np.float32)
    for var_name, val in x_values.items():
        # var_name 格式如 "x[0,0,1]"，在 mi['var_names'] 中查找
        if var_name in var_names:
            idx = var_names.index(var_name)
            y[idx] = float(val)

    # 6. 变量掩码：只对 x 变量计算损失
    var_mask = (is_target > 0.5)

    # 7. 组装 Data 对象
    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        y=torch.from_numpy(y).float(),
        var_mask=torch.from_numpy(var_mask).bool(),
        num_vars=V,
        num_constrs=C,
        num_nodes=total_nodes,
    )
    data.var_names = var_names  # 非 tensor 属性，保留用于 warmstart 映射

    return data
