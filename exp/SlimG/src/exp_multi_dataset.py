import os
import sys
import time
from copy import deepcopy
import pandas as pd
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import random

# 添加matplotlib绘图模块
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_data_multi import load_multi_dataset
from train_multi import train_slimG_with_grid_search, set_seed
from models import load_model

def plot_loss_with_acc(loss_history, val_acc_history):
    # 绘制loss和accuracy曲线，与main.py保持一致
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    
    # 处理tensor格式
    if len(loss_history) > 0 and isinstance(loss_history[0], torch.Tensor):
        loss_history = [loss.detach().numpy() for loss in loss_history]
    if len(val_acc_history) > 0 and isinstance(val_acc_history[0], torch.Tensor):
        val_acc_history = [acc.detach().numpy() for acc in val_acc_history]
    
    # 绘制loss曲线
    ax1.plot(range(len(loss_history)), loss_history, c='blue')
    ax1.set_ylabel('Loss', color='blue', fontsize=14)
    ax1.tick_params(axis='y', colors='blue', labelsize=12)

    # 绘制accuracy曲线
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history, c='red')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Accuracy', color='red', fontsize=14)
    ax2.tick_params(axis='y', colors='red', labelsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=" * 100)
    print("开始多数据集SlimG模型实验 - SlimG模型在每个数据集上运行5次独立实验")
    print("=" * 100)
    
    # 定义要测试的数据集后缀 (与TDHNN保持一致)
    dataset_suffixes = [0]
    
    # 检测设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 存储所有数据集的结果
    all_datasets_results = {}
    
    # 创建结果保存文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"SlimG_experiment_results_{timestamp}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("SlimG模型多数据集实验结果\n")
        f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用设备: {device}\n")
        f.write("=" * 100 + "\n\n")
    
    for suffix in dataset_suffixes:
        print(f"\n{'#'*50} 数据集后缀: {suffix} {'#'*50}")
        
        # 存储当前数据集的结果
        dataset_results = {}
        dataset_info = None
        
        print(f"\n{'='*25} 数据集_{suffix} - 模型: SlimG {'='*25}")
        
        test_acc_history = []
        model_parameters_info = []
        
        for trial in range(5):
            print(f"\n{'-'*15} 数据集_{suffix} - SlimG - 实验 {trial+1}/5 {'-'*15}")
            
            # 设置随机种子
            set_seed(trial)
            
            try:
                # 加载数据
                data = load_multi_dataset(suffix, device=device)
                
                # 只在第一次实验时记录数据集信息
                if trial == 0:
                    dataset_info = {
                        'suffix': suffix,
                        'features_shape': data['features'].shape,
                        'labels_shape': data['labels'].shape,
                        'num_classes': data['num_classes'],
                        'train_size': len(data['train_idx']),
                        'val_size': len(data['val_idx']),
                        'test_size': len(data['test_idx'])
                    }
                    
                    print(f"数据集_{suffix}信息:")
                    print(f"特征矩阵X形状: {data['features'].shape}")
                    print(f"标签矩阵形状: {data['labels'].shape}, 类别数: {data['num_classes']}")
                    print(f"训练集样本数: {len(data['train_idx'])}")
                    print(f"验证集样本数: {len(data['val_idx'])}")
                    print(f"测试集样本数: {len(data['test_idx'])}")
                    print(f"边索引形状: {data['edge_index'].shape}")
            
            except FileNotFoundError as e:
                print(f"文件未找到: {e}")
                print(f"跳过数据集_{suffix}")
                break
            except Exception as e:
                print(f"加载数据时出错: {e}")
                print(f"跳过数据集_{suffix}")
                break
            
            # 训练模型
            print(f"开始训练 SlimG...")
            
            try:
                # 调用训练函数
                best_acc, val_acc_history, loss_history = train_slimG_with_grid_search(
                    data, max_epochs=500, lr=0.01, weight_decay=5e-4, verbose=False
                )
                
                # 只在第一次实验时打印模型信息
                if trial == 0:
                    # 创建模型用于参数统计
                    model = load_model(
                        num_nodes=data['num_nodes'],
                        num_features=data['num_features'],
                        num_classes=data['num_classes']
                    ).to(device)
                    
                    # 预处理以初始化所有层
                    model.preprocess(data['features'], data['edge_index'], data['labels'], device)
                    
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"SlimG 模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
                    
                    # 存储模型参数信息
                    model_info = {
                        "model": "SlimG",
                        "parameters": total_params,
                        "accuracy": best_acc
                    }
                    model_parameters_info.append(model_info)
                
                print(f"数据集_{suffix} - SlimG 训练完成! 最佳验证准确率: {best_acc:.5f}")
                
                # 绘制损失和准确率曲线（不显示图形）
                plot_loss_with_acc(loss_history, val_acc_history)
                
                print(f"数据集_{suffix} - SlimG 实验{trial+1}最终验证准确率: {best_acc:.5f}")
                test_acc_history.append(best_acc)
                
            except Exception as e:
                print(f"训练过程中出错: {e}")
                print(f"跳过当前实验")
                import traceback
                traceback.print_exc()
                continue
        
        # 如果成功完成了实验，计算统计结果
        if test_acc_history:
            # 计算统计结果
            mean_acc = np.mean(test_acc_history)
            std_acc = np.std(test_acc_history)
            
            print(f"\n数据集_{suffix} - SlimG 模型结果统计:")
            print(f"各次实验结果: {[f'{acc:.4f}' for acc in test_acc_history]}")
            print(f"平均验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"最高准确率: {max(test_acc_history):.4f}")
            print(f"最低准确率: {min(test_acc_history):.4f}")
            
            # 存储结果
            dataset_results["SlimG"] = {
                'accuracy_list': test_acc_history,
                'mean': mean_acc,
                'std': std_acc,
                'max': max(test_acc_history),
                'min': min(test_acc_history),
                'parameters': model_parameters_info[0]['parameters'] if model_parameters_info else 0
            }
            
            # 存储到总结果中
            all_datasets_results[suffix] = {
                'dataset_info': dataset_info,
                'model_results': dataset_results
            }
            
            # 显示当前数据集的SlimG模型结果
            print(f"\n{'='*30} 数据集_{suffix} SlimG模型结果 {'='*30}")
            
            for model_name, results in dataset_results.items():
                print(f"SlimG模型统计:")
                print(f"平均准确率: {results['mean']:.4f} ± {results['std']:.4f}")
                print(f"最高准确率: {results['max']:.4f}")
                print(f"参数数量: {results['parameters']:,}")
            
            # 保存到文件
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n数据集_{suffix} 结果:\n")
                f.write("=" * 50 + "\n")
                if dataset_info:
                    f.write(f"数据集信息: 特征形状{dataset_info['features_shape']}, 类别数{dataset_info['num_classes']}, ")
                    f.write(f"训练集{dataset_info['train_size']}, 验证集{dataset_info['val_size']}, 测试集{dataset_info['test_size']}\n")
                
                for model_name, results in dataset_results.items():
                    f.write(f"SlimG模型结果:\n")
                    f.write(f"平均准确率: {results['mean']:.4f} ± {results['std']:.4f}\n")
                    f.write(f"最高准确率: {results['max']:.4f}\n")
                    f.write(f"参数数量: {results['parameters']:,}\n")
                    f.write(f"各次实验: {results['accuracy_list']}\n")
                f.write("\n" + "="*80 + "\n\n")
    
    # 最终汇总所有结果
    print("\n" + "#" * 100)
    print("所有数据集上SlimG模型的最终对比结果")
    print("#" * 100)
    
    # 创建汇总表格
    summary_results = []
    for suffix, data in all_datasets_results.items():
        for model_name, results in data['model_results'].items():
            summary_results.append({
                'dataset': f"数据集_{suffix}",
                'mean_acc': results['mean'],
                'std_acc': results['std'],
                'max_acc': results['max'],
                'parameters': results['parameters']
            })
    
    # 按平均准确率排序
    summary_results.sort(key=lambda x: x['mean_acc'], reverse=True)
    
    print(f"{'数据集':<15} {'平均准确率':<12} {'标准差':<8} {'最高准确率':<12} {'参数数量':<12}")
    print("-" * 70)
    for result in summary_results:
        print(f"{result['dataset']:<15} {result['mean_acc']:.4f}       {result['std_acc']:.4f}   {result['max_acc']:.4f}       {result['parameters']:,}")
    
    # 保存最终汇总结果
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "#" * 100 + "\n")
        f.write("最终汇总结果 - SlimG在各数据集上的性能（按平均准确率排序）\n")
        f.write("#" * 100 + "\n")
        f.write(f"{'数据集':<15} {'平均准确率':<12} {'标准差':<8} {'最高准确率':<12} {'参数数量':<12}\n")
        f.write("-" * 70 + "\n")
        for result in summary_results:
            f.write(f"{result['dataset']:<15} {result['mean_acc']:.4f}       {result['std_acc']:.4f}   {result['max_acc']:.4f}       {result['parameters']:,}\n")
        
        f.write(f"\n实验完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 找出最佳数据集结果
    if summary_results:
        best_overall = summary_results[0]
        print(f"\nSlimG最佳表现: {best_overall['dataset']} (平均准确率: {best_overall['mean_acc']:.4f})")
    
    print(f"\n所有实验结果已保存到文件: {results_file}")
    print("实验完成！") 