import torch
from dhg.models import HGNN, HGNNP, HNHN, HyperGCN, GCN, GAT, GIN

# 假设的输入维度和输出类别数
n_features = 1024  # 特征维度，可以根据实际情况调整
class_num = 7  # 类别数量，可以根据实际情况调整

# 初始化所有模型（包括超图神经网络模型和普通图神经网络模型）
models = {
    # 超图神经网络模型
    "HGNN": HGNN(n_features, 64, class_num),
    "HGNNP": HGNNP(n_features, 64, class_num, use_bn=True),
    "HNHN": HNHN(n_features, 64, class_num, use_bn=True),
    "HyperGCN": HyperGCN(n_features, 64, class_num, use_bn=True),
    
    # 普通图神经网络模型（保持与原始脚本一致）
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