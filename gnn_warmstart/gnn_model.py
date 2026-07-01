"""
GNN 模型：二分图消息传递（Bipartite GNN）用于 MILP 变量预测

架构：
  1. 变量编码器 + 约束编码器 → 投影到统一隐藏维度
  2. K 层 BipartiteConv（每层: V→C 聚合 → C 更新, C→V 聚合 → V 更新）
  3. 分类器 → 每个变量节点的 logit
"""
import torch
import torch.nn as nn


class BipartiteConv(nn.Module):
    """
    一轮二分图消息传递：
      1. 变量 → 约束：聚合邻居变量特征到约束节点
      2. 约束 → 变量：聚合邻居约束特征到变量节点
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # V → C: 变量发消息给约束
        self.msg_v2c = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # h_v + edge_attr
            nn.ReLU(),
        )
        self.upd_c = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # h_c_old + agg_msg
            nn.ReLU(),
        )

        # C → V: 约束发消息给变量
        self.msg_c2v = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
        )
        self.upd_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, h_v, h_c, edge_index_v2c, edge_attr):
        """
        Args:
            h_v: [V, D] 变量节点嵌入 (局部索引 0..V-1)
            h_c: [C, D] 约束节点嵌入 (局部索引 0..C-1)
            edge_index_v2c: [2, E] 边索引，src=变量(全局0..V-1)，dst=约束(全局V..V+C-1)
            edge_attr: [E, 1] 边特征
        Returns:
            h_v_new: [V, D], h_c_new: [C, D]
        """
        V = h_v.size(0)
        C = h_c.size(0)
        D = h_v.size(1)

        # 将约束的全局索引映射回局部 0..C-1
        dst_local = edge_index_v2c[1] - V   # V+C-1 → C-1

        # ——— V → C ———
        src_v = h_v[edge_index_v2c[0]]                        # [E, D]  src 已是 0..V-1
        msg_input = torch.cat([src_v, edge_attr], dim=-1)     # [E, D+1]
        msgs = self.msg_v2c(msg_input)                         # [E, D]

        agg_c = torch.zeros(C, D, device=h_c.device)
        idx = dst_local.unsqueeze(-1).expand(-1, D)
        agg_c = agg_c.scatter_add(0, idx, msgs)

        h_c_new = self.upd_c(torch.cat([h_c, agg_c], dim=-1))  # [C, D]

        # ——— C → V ———
        # 翻转边：局部约束索引(0..C-1)→src, 变量索引(0..V-1)→dst
        src_c = h_c_new[dst_local]                             # [E, D]
        msg_input = torch.cat([src_c, edge_attr], dim=-1)
        msgs = self.msg_c2v(msg_input)                          # [E, D]

        agg_v = torch.zeros(V, D, device=h_v.device)
        idx = edge_index_v2c[0].unsqueeze(-1).expand(-1, D)   # 变量已在 0..V-1
        agg_v = agg_v.scatter_add(0, idx, msgs)

        h_v_new = self.upd_v(torch.cat([h_v, agg_v], dim=-1))  # [V, D]

        return h_v_new, h_c_new


class WarmStartGNN(nn.Module):
    """
    MILP 热启动 GNN 模型

    输入：二分图（变量节点 + 约束节点 + 带系数的边）
    输出：每个变量节点的 logit（> 0 表示预测为 1）
    """

    def __init__(self, var_feat_dim=7, constr_feat_dim=4,
                 hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 初始编码：将变量和约束特征投影到相同隐藏维度
        self.var_encoder = nn.Sequential(
            nn.Linear(var_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.constr_encoder = nn.Sequential(
            nn.Linear(constr_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # 消息传递层
        self.convs = nn.ModuleList([
            BipartiteConv(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 变量二值分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data 对象，包含:
                - x: [V+C, feat_dim] 全部节点特征
                - edge_index: [2, 2E] 双向边索引
                - edge_attr: [2E, 1] 边特征
                - num_vars: int 变量节点数
        Returns:
            logits: [V] 每个变量节点的二值 logit
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        V = data.num_vars

        # 提取变量和约束的特征（前 7 维是变量特征，约束特征取前 4 维）
        var_feat = x[:V, :7]
        constr_feat = x[V:, :4] if x.size(0) > V else torch.zeros(0, 4, device=x.device)

        # 编码
        h_v = self.var_encoder(var_feat)         # [V, D]
        h_c = self.constr_encoder(constr_feat)   # [C, D]

        # 构造 V→C 方向边（从正向边中筛选，src < V 即为 v→c）
        is_v2c = (edge_index[0] < V)
        e_v2c = edge_index[:, is_v2c]
        e_attr = edge_attr[is_v2c]

        # 批量归一化边特征
        if e_attr.numel() > 0:
            e_std = e_attr.std()
            if e_std > 1e-8:
                e_attr = (e_attr - e_attr.mean()) / e_std

        # 消息传递
        for conv in self.convs:
            h_v, h_c = conv(h_v, h_c, e_v2c, e_attr)
            h_v = self.dropout(h_v)

        # 分类
        logits = self.classifier(h_v).squeeze(-1)  # [V]

        return logits

    @torch.no_grad()
    def predict(self, data, threshold=0.5):
        """推理：返回预测的二进制值"""
        self.eval()
        logits = self.forward(data)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()
        return preds, probs
