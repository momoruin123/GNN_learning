"""
GNN 热启动训练配置
"""

# ========== 数据配置 ==========
TRAIN_DATA_DIR = "./training_data"  # 训练数据目录（data_generation 输出）
MODEL_SAVE_DIR = "./models"         # 模型保存目录

# ========== 训练参数 ==========
NUM_EPOCHS = 100                     # 训练轮数
LEARNING_RATE = 3e-4                # 学习率
WEIGHT_DECAY = 1e-5                 # 权重衰减
TRAIN_RATIO = 0.7                   # 训练集比例
VAL_RATIO = 0.15                    # 验证集比例
TEST_RATIO = 0.15                   # 测试集比例

# ========== GNN 模型参数 ==========
HIDDEN_DIM = 128                     # 隐藏层维度
NUM_CONV_LAYERS = 4                 # 消息传递层数（bipartite conv rounds）
DROPOUT = 0.1                       # Dropout 概率

# ========== 训练设施 ==========
BATCH_SIZE = 1                      # 批量大小（图数量，含分图结构需逐图处理）
GRAD_ACCUM_STEPS = 4               # 梯度累积步数（等效 batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS）
NUM_WORKERS = 0                     # DataLoader 工作进程数（Windows 建议 0）
DEVICE = "cpu"                      # 训练设备 ("cpu" 或 "cuda")

# ========== 热启动参数 ==========
PREDICTION_THRESHOLD = 0.05            # 分类阈值（带 pos_weight 训练后正样本 sigmoid 很低）
