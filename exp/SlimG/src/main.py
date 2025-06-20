import argparse
import io
from distutils.util import strtobool
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch import nn, optim

import utils
from data import load_data, split_nodes
from models import load_model
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def set_seed(seed: int):
    r"""Set the random seed of DHG.

    .. note::
        When you call this function, the random seeds of ``random``, ``numpy``, and ``pytorch`` will be set, simultaneously.

    Args:
        ``seed`` (``int``): The specified random seed.
    """
    global _MANUAL_SEED
    _MANUAL_SEED = seed
    random.seed(_MANUAL_SEED)
    np.random.seed(_MANUAL_SEED)
    torch.manual_seed(_MANUAL_SEED)


def plot_loss_with_acc(loss_history, val_acc_history):
    # loss_history.detach().numpy()
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    if isinstance(loss_history[0], torch.Tensor):
        loss_history = [loss.detach().numpy() for loss in loss_history]
    if isinstance(val_acc_history[0], torch.Tensor):
        val_acc_history = [acc.detach().numpy() for acc in val_acc_history]
    ax1.plot(range(len(loss_history)), loss_history,
             c='blue')
    # plt.ylabel('Loss')
    ax1.set_ylabel('Loss', color='blue', fontsize=14)  # 设置竖轴颜色为橙色
    ax1.tick_params(axis='y', colors='blue', labelsize=12)  # 设置竖轴刻度颜色为橙色

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c='red')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    # plt.ylabel('ValAcc')
    ax2.set_ylabel('Accuracy', color='red', fontsize=14)  # 设置竖轴颜色为蓝色
    ax2.tick_params(axis='y', colors='red', labelsize=12)  # 设置竖轴刻度颜色为蓝色
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # 调整子图之间的间距
    plt.show()

def parse_args():
    def str2bool(x):
        return bool(strtobool(x))

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=f'{utils.ROOT}/out/temp')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000))
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=500)
    return parser.parse_args()


def to_regularizer(model, lambda_1, lambda_2):

    out = []
    for j, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and 'bias' not in name:
            out.append(lambda_1 * torch.abs(param).sum())
            out.append(lambda_2 * torch.sqrt(torch.pow(param, 2).sum()))
    return torch.stack(out).sum()


def train_model(args, model, features, labels, edge_index, trn_nodes,
                val_nodes,test_nodes, lambda_1, lambda_2):
    loss_history = []
    val_acc_history = []
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # @torch.no_grad()
    def evaluate(nodes):
        model.eval()
        pred_ = model(features, edge_index)
        # print(pred_.shape)
        # print(labels.shape)
        # labels_ = labels[nodes]
        # pred_ = torch.tensor(pred_)
        pred_ = pred_[nodes]
        # print(nodes.dtype)
        # print(pred_.dtype)
        # pred_ = [pred_[i] for i in nodes]
        labels_ = labels[nodes]
        lbls = labels_.to('cpu')
        lbls_ = lbls.cpu().numpy()
        outputs = pred_.cpu()
        preds = np.where(F.sigmoid(outputs) > 0.5, 1, 0)
        labels_ = labels_.float()
        loss_ = loss_func(pred_, labels_).item()
        # acc_ = (pred_.argmax(dim=1) == labels_).float().mean().item()
        TP = sum((lbls_ == 1) & (preds == 1))
        FP = sum((lbls_ == 0) & (preds == 1))
        FN = sum((lbls_ == 1) & (preds == 0))
        TN = sum((lbls_ == 0) & (preds == 0))
        TP = TP.sum().item()
        FP = FP.sum().item()
        FN = FN.sum().item()
        TN = TN.sum().item()
        # 根据上面得到的值计算 Accuracy
        acc_ = (TP + TN) / (TP + FP + FN + TN)

        return loss_, acc_



    logs = []
    best_epoch, best_acc, best_model = -1, 0, io.BytesIO()
    best_loss = np.inf
    for epoch in range(500):
        model.train()
        # optimizer.step(closure)
        optimizer.zero_grad()
        pred_ = model(features, edge_index)
        # labels_ = labels[trn_nodes]
        pred_, labels_ = pred_[trn_nodes], labels[trn_nodes]
        labels_ = labels_.float()
        loss1 = loss_func(pred_, labels_)
        if lambda_1 > 0 or lambda_2 > 0:
            loss2 = to_regularizer(model, lambda_1, lambda_2)
        else:
            loss2 = 0

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        val_loss, val_acc = evaluate(val_nodes)
        val_acc_history.append(val_acc)
        loss_history.append(loss)
        if args.verbose:
            print(logs[-1])
        # print(epoch)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            # print(best_acc)
            from copy import deepcopy
            best_state = deepcopy(model.state_dict())
    # model.eval()
    # model.load_state_dict(best_model)
    # test_loss,test_acc = evaluate(test_nodes)
    # print('test_acc', test_acc)
    # plot_loss_with_acc(loss_history, val_acc_history)
    # acc_arr.append(test_acc)
        # elif epoch >= best_epoch + args.patience:
        #     break


    return best_epoch, best_acc, best_state, val_acc_history, loss_history


def main():
    args = parse_args()
    utils.set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    ### Homophily graphs
    datasets = ['cora']

     ### Heterophily graphs
    # datasets = ['chameleon', 'squirrel', 'actor', 'penn94', 'twitch', 'pokec']

     ### Synthetic graphs
    # datasets = ['synthetic-semantic-uniform-individual',
    #             'synthetic-random-clustered-homophily', 'synthetic-random-bipartite-heterophily', 
    #             'synthetic-structural-clustered-homophily', 'synthetic-structural-bipartite-heterophily',
    #             'synthetic-semantic-clustered-homophily', 'synthetic-semantic-bipartite-heterophily']

    for dataset in datasets:
        print("dataset")
        #
        X = pd.read_csv('G:\CODE\myHGNN\data\data\X.csv', header=None, skiprows=1)
        X = np.array(X)
        H = torch.tensor(X)
        features = torch.tensor(X)
        lbl = pd.read_csv('G:\CODE\myHGNN\data\data\Y.csv', header=None, skiprows=1)
        lbl = np.array(lbl)
        labels = torch.tensor(lbl)
        class_num = int(lbl.shape[1])
        idx = pd.read_csv('G:\CODE\myHGNN\data\data\idx.csv', header=None, skiprows=1)
        idx = np.array(idx)
        train_mask = np.where(idx == 1)[0]  # 训练集索引
        test_mask = np.where(idx == 2)[0]
        val_mask = np.where(idx == 0)[0]
        with open('G:\CODE\myHGNN\data\data\edge_A1.pkl', 'rb') as f1:
            data1 = pickle.load(f1)
        # 读取第二个 pkl 文件
        with open('G:\CODE\myHGNN\data\data\edge_B1.pkl', 'rb') as f2:
            data2 = pickle.load(f2)
        # 合并两个对象，每个对象作为一个列表元素，组成一个新的列表
        tensor1 = torch.tensor(data1)
        tensor2 = torch.tensor(data2)
        # from torch_cluster import knn_graph
        # edge_index=knn_graph(H,k=15)
        # 合并两个张量，使得每一行对应原来两个对象的一行数据
        edge_index = torch.stack([tensor1, tensor2])
        # with open('/home/data0/xuweize/myHGNN/data/H_test.pkl', 'rb') as f3:
        #     edge_index = pickle.load(f3)
        # features, edge_index, labels = load_data(dataset)
        features, edge_index, labels = features.float().to(device), edge_index.to(device), labels.to(device)
        print(edge_index.shape)
        print(dataset)
        print('# of Nodes:', len(labels))
        print('# of Classes:', class_num)
        print('# of Features:', features.shape)
        print()

        def evaluate(nodes):
            model.eval()
            pred_ = model(features, edge_index)[nodes]
            labels_ = labels[nodes]
            lbls = labels_.to(device)
            lbls_ = lbls.cpu().numpy()
            outputs = pred_.cpu()
            preds = np.where(F.sigmoid(outputs) > 0.5, 1, 0)
            TP = sum((lbls_ == 1) & (preds == 1))
            FP = sum((lbls_ == 0) & (preds == 1))
            FN = sum((lbls_ == 1) & (preds == 0))
            TN = sum((lbls_ == 0) & (preds == 0))
            TP = TP.sum().item()
            FP = FP.sum().item()
            FN = FN.sum().item()
            TN = TN.sum().item()
            # 根据上面得到的值计算 Accuracy
            acc_ = (TP + TN) / (TP + FP + FN + TN)
            return acc_

        model = load_model(
                num_nodes=features.size(0),
                num_features=features.size(1),
                num_classes=class_num
            ).to(device)
        model.preprocess(features, edge_index, labels, device)
        
        # 添加参数统计信息
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
        
        # 模型大小估计 (MB)
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设每个参数是float32 (4字节)
        print(f"\n模型大小(MB): {model_size_mb:.2f}")
        
        model.feature_size()

        def grid_search(args, model, features, labels, trn_nodes, val_nodes,test_nodes):
            search_range_1 = 0.8e-3
            search_range_2 = 0.8e-4

            val_acc_dict = {}
            # for lambda_1 in search_range_1:
            #     for lambda_2 in search_range_2:
            model.reset_parameters()
            _, acc, _, loss_history, val_acc_history= train_model(
                    args=args,
                    model=model,
                    features=features,
                    labels=labels,
                    edge_index=edge_index,
                    trn_nodes=trn_nodes,
                    val_nodes=val_nodes,
                    test_nodes=test_nodes,
                    lambda_1=search_range_1,
                    lambda_2=search_range_2)
            val_acc_dict[(search_range_1, search_range_2)] = acc

            return max(val_acc_dict, key=val_acc_dict.get)

        acc_arr = []
        for i in tqdm(range(5)):
            print("这是i",i)
            trn_nodes = train_mask
            val_nodes = val_mask
            test_nodes = test_mask
            set_seed(random.randint(0, 10000))
            # if dataset[:9] == 'synthetic':
            #     trn_nodes, val_nodes, test_nodes = split_nodes(labels.cpu(), ratio=(0.6, 0.2, 0.2), seed=i)
            # else:
            #     trn_nodes, val_nodes, test_nodes = split_nodes(labels.cpu(), ratio=(0.025, 0.025, 0.95), seed=i)

            best_parm = grid_search(args, model, features, labels, trn_nodes, val_nodes,test_nodes)

            model.reset_parameters()
            epoch, acc, best_model, val_acc_history, loss_history = train_model(
                    args=args,
                    model=model,
                    features=features,
                    labels=labels,
                    edge_index=edge_index,
                    trn_nodes=trn_nodes,
                    val_nodes=val_nodes,
                    test_nodes=test_nodes,
                    lambda_1=best_parm[0],
                    lambda_2=best_parm[1])
            # net.load_state_dict(best_state)
            model.load_state_dict(best_model)
            test_acc = evaluate(test_nodes)
            print('test_acc',test_acc)
            plot_loss_with_acc(loss_history, val_acc_history)
            acc_arr.append(test_acc)

        print('%.1f +- %.1f' % (np.mean(acc_arr) * 100, np.std(acc_arr) * 100))
        print()

if __name__ == '__main__':
    main()
