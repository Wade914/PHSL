import os.path as osp
import torch
import numpy as np
import pandas as pd
import pickle


def load_multi_dataset(suffix=None, device='cuda'):
    """
    加载多数据集版本的数据
    
    Args:
        suffix: 数据集后缀，如 10, 5, 0, -5 等
        device: 设备类型
    
    Returns:
        data: 包含特征、标签、索引等的字典
    """
    # 根据suffix参数构建文件路径
    if suffix is not None:
        suffix_str = str(suffix)
    else:
        suffix_str = "10"  # 默认值，保持向后兼容
    
    # 构建相对路径，向上两级找到exp目录
    base_path = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    data_path = osp.join(base_path, 'data')
    
    # 加载特征数据
    fts = pd.read_csv(osp.join(data_path, f'X_{suffix_str}.csv'), header=None, skiprows=1)
    fts = np.array(fts)

    # 加载标签数据
    lbl = pd.read_csv(osp.join(data_path, f'Y_{suffix_str}.csv'), header=None, skiprows=1)
    lbls = np.array(lbl)
    class_num = int(lbl.shape[1])
    
    # 加载索引数据
    idx = pd.read_csv(osp.join(data_path, f'idx_{suffix_str}.csv'), header=None, skiprows=1)
    idx = np.array(idx)
    
    # 获取训练、验证、测试集索引
    train_idx = np.where(idx == 1)[0]  # 训练集索引
    test_idx = np.where(idx == 2)[0]   # 测试集索引
    val_idx = np.where(idx == 0)[0]    # 验证集索引
    
    # 加载边数据
    try:
        # 尝试加载对应后缀的边文件（格式：edge_A1_{suffix}.pkl）
        with open(osp.join(data_path, f'edge_A1_{suffix_str}.pkl'), 'rb') as f1:
            data1 = pickle.load(f1)
        with open(osp.join(data_path, f'edge_B1_{suffix_str}.pkl'), 'rb') as f2:
            data2 = pickle.load(f2)
    except FileNotFoundError:
        # 如果没有对应后缀的边文件，使用默认的边文件
        try:
            with open(osp.join(data_path, 'edge_A1.pkl'), 'rb') as f1:
                data1 = pickle.load(f1)
            with open(osp.join(data_path, 'edge_B1.pkl'), 'rb') as f2:
                data2 = pickle.load(f2)
        except FileNotFoundError:
            # 直接抛出错误，不自动创建KNN图
            raise FileNotFoundError(
                f"边文件不存在！请准备以下文件之一：\n"
                f"1. {osp.join(data_path, f'edge_A1_{suffix_str}.pkl')} 和 {osp.join(data_path, f'edge_B1_{suffix_str}.pkl')}\n"
                f"2. {osp.join(data_path, 'edge_A1.pkl')} 和 {osp.join(data_path, 'edge_B1.pkl')}"
            )
    
    # 转换为PyTorch张量
    device_obj = torch.device(device)
    
    features = torch.tensor(fts).float().to(device_obj)
    labels = torch.tensor(lbls).float().to(device_obj)  # 多标签分类使用float
    
    tensor1 = torch.tensor(data1)
    tensor2 = torch.tensor(data2)
    edge_index = torch.stack([tensor1, tensor2]).to(device_obj)
    
    train_idx = torch.tensor(train_idx).long()
    test_idx = torch.tensor(test_idx).long()
    val_idx = torch.tensor(val_idx).long()
    
    data = {
        'features': features,
        'labels': labels,
        'edge_index': edge_index,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'val_idx': val_idx,
        'num_classes': class_num,
        'num_features': features.shape[1],
        'num_nodes': features.shape[0]
    }
    
    return data 