#!/usr/bin/env python3
"""
SlimG 多数据集实验启动脚本

运行方法:
python run_experiment.py

此脚本将在7个不同的数据集上测试SlimG模型，
每个数据集运行5次独立实验，并生成详细的性能报告。
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 运行主实验
if __name__ == "__main__":
    print("启动SlimG多数据集实验...")
    print("=" * 60)
    print("实验配置:")
    print("- 数据集后缀: 0, -5, -10, -20, -30, -40, -50")
    print("- 模型: SlimG")
    print("- 每个数据集运行5次独立实验")
    print("- 总计实验次数: 7 × 5 = 35 次实验")
    print("=" * 60)
    
    try:
        # 导入并运行实验
        from exp_multi_dataset import *
        print("实验开始...")
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有必要的模块都已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        print("请检查数据文件是否存在，模型是否正确配置")
        sys.exit(1) 