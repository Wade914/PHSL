
import torch
from models import load_model
import numpy as np

# 假设的输入维度和输出类别数
num_nodes = 1759  # 节点数量
num_features = 1024  # 特征维度
num_classes = 7  # 类别数量

# 强制使用CPU以避免CUDA相关错误
device = torch.device('cpu')

# 初始化模型
model = load_model(
    num_nodes=num_nodes,
    num_features=num_features,
    num_classes=num_classes
)

# 创建虚拟数据用于初始化模型 - 使用CPU
x = torch.rand(num_nodes, num_features)
edge_index = torch.randint(0, num_nodes, (2, 5000))
labels = torch.randint(0, num_classes, (num_nodes, num_classes)).float()

# 调用预处理函数初始化模型参数
model.preprocess(x, edge_index, labels, device)

# 计算并打印模型参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nSlimG 模型摘要:")
print(f"总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# 打印各个组件的参数数量
print("\n各组件参数数量:")
print(f"lin1 (结构特征): {sum(p.numel() for p in model.lin1.parameters()):,}")
print(f"lin2 (原始特征): {sum(p.numel() for p in model.lin2.parameters()):,}")

for i, l in enumerate(model.linears):
    print(f"linear_{i} (传播特征): {sum(p.numel() for p in l.parameters()):,}") 