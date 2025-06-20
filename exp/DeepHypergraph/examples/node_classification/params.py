import os
# 设置环境变量，必须在导入numpy和torch之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from dhg.models import GCN, GAT, GIN

# 假设的输入维度和输出类别数
n_features = 512  # 特征维度，可以根据实际情况调整
class_num = 7  # 类别数量，可以根据实际情况调整

# 初始化模型
models = {
    "GCN": GCN(n_features, 64, class_num),
    "GAT": GAT(n_features, 64, class_num, 4, use_bn=True),
    "GIN": GIN(n_features, 64, class_num, 2)
}

# 计算并打印每个模型的参数
print("\n模型参数比较:")
for name, model in models.items():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{name} 模型摘要:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 打印模型结构（可选）
    print(f"模型结构:")
    print(model)