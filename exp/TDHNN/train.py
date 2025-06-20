import torch
from networks import HGNN_classifier,GCN,MLP,GAT
import torch.nn.functional as F
import random
import numpy as np
import time
import datetime
import torch.nn as nn
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def contrast_loss(H_raw,mask,labels):               # Nodes with same label should belong to same edge
    # lbl_num = labels.shape[1]
    label_mapping = [
        [1, 0, 0, 0, 0],  # normal
        [0, 1, 0, 0, 0],  # fault1
        [0, 0, 1, 0, 0],  # fault2
        [0, 0, 0, 1, 0],  # fault3
        [0, 0, 1, 0, 1],  # fault4, fault2
        [0, 1, 0, 0, 1],  # fault4, fault1
        [0, 1, 0, 1, 0]  # fault1, fault3
    ]
    total_loss = 0
    # print(H_raw)
    for h in H_raw:
        # print(h.shape)
        for lbl in label_mapping:
            lbl = torch.tensor(lbl).cuda()
            # print(lbl.device)
            # print(labels.device)
            lbl_mask = (labels == lbl).all(dim=1)
            # print(h[mask].shape)
            # print(lbl_mask.shape)
            src = h[mask][lbl_mask]
            target_idx = [i for i in range(src.shape[0])]
            random.shuffle(target_idx)
            target = src[target_idx]
            loss = F.mse_loss(src,target)
            total_loss = total_loss + loss
    return total_loss

def contrast_loss2(H, x):                         # Nodes in same edge should have similar features
    total_loss = 0
    feartures = x
    for h in H:
        cc = h.ceil().abs()
        for i in range(cc.shape[1]):               # h is n*n dimension
            col_mask = cc[:,i] == 1
            src = feartures[col_mask]             # Nodes belonging to same edge
            target_idx = [i for i in range(src.shape[0])]
            random.shuffle(target_idx)
            target = src[target_idx]                # Randomly shuffle idx
            loss = F.mse_loss(src,target) + 1e-8           # Nodes in same edge should have similar features
            if loss > 1e-8:
                total_loss = total_loss + loss
    return total_loss

def laplacian_rank(H_list, device):
    # L = I - Dv^(-1/2) W De^(-1) H^(T) Dv^(-1/2)
    rank_list = []
    for tmpH in H_list:
        H = tmpH.clone()

        ## Delete empty edges
        n_edge = H.shape[1]
        tmp_sum = H.sum(dim=0)
        index = []
        for i in range(n_edge):
            if tmp_sum[i] != 0:
                index.append(i)

        H = H[:, index]
        ################
        n_node = H.shape[0]
        n_edge = H.shape[1]

        # the weight of the hyperedge
        # W = np.ones(n_edge)
        W = torch.ones(n_edge).to(device)

        # the degree of the node
        # DV = np.sum(H * W, axis=1)
        DV = torch.sum(H * W, axis=1)

        # the degree of the hyperedge
        # DE = np.sum(H, axis=0)
        DE = torch.sum(H, axis=0)

        # invDE = np.mat(np.diag(np.power(DE, -1)))
        invDE = torch.diag(torch.pow(DE,-1))

        # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        DV2 = torch.diag(torch.pow(DV, -0.5))

        # W = np.mat(np.diag(W))
        # H = np.mat(H)
        HT = H.T

        I = torch.eye(n_node, n_node).to(device)

        L = I - DV2 @ H @ W @ invDE @ HT @ DV2

        rank_L = torch.linalg.matrix_rank(L)

        rank_list.append(rank_L)

    print("===========================> Rank of L is: ", rank_list)


def train(model, optimizer, data, args):
    device = torch.device(args.device)
    model.to(device)
    train_mask = data['train_idx']
    labels = data['lbls'][train_mask]

    best_acc = 0
    patience = 0
    best_epoch = 0
    loss_history = []
    val_acc_history = []
    spent_time = []

    for epoch in range(args.epoch):
        t0 = time.time()

        model.train()
        optimizer.zero_grad()

        args.stage = 'train'
        out, x, H, H_raw = model(data,args)
        # print(H)
        # print(H_raw)
        contra_ls = 0
        contra_ls2 = 0
        if H_raw is not None:
            # if epoch % 20 == 0 or epoch == args.epoch - 1:
            #     laplacian_rank(H, args.device)
                
            contra_ls = contrast_loss(H_raw,train_mask,labels)
            contra_ls2 = contrast_loss2(H, x)
        # print(out.dtype)
        labels = labels.float()
        # print(labels.dtype)
        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(out[train_mask], labels) + contra_ls * args.namuda + contra_ls2 * (args.namuda2 / 1000)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        # _, pred = pred[train_mask].max(dim=1)
        # correct = int(pred.eq(labels).sum().item())
        # acc = correct / len(labels)
        outs, lbls = out[train_mask], labels
        lbls_ = lbls.cpu()
        outputs = outs.cpu()
        preds = np.where(F.sigmoid(outputs) > 0.5, 1, 0)
        # res = evaluator.validate(lbls_, outs)
        TP = sum((lbls_ == 1) & (preds == 1))
        FP = sum((lbls_ == 0) & (preds == 1))
        FN = sum((lbls_ == 1) & (preds == 0))
        TN = sum((lbls_ == 0) & (preds == 0))
        TP = TP.sum().item()
        FP = FP.sum().item()
        FN = FN.sum().item()
        TN = TN.sum().item()
        # Calculate accuracy based on above values
        acc = (TP + TN) / (TP + FP + FN + TN)
        # val_acc, val_loss = evaluate(model, data, stage = 'val')
        args.stage = 'test'
        test_acc, test_loss = evaluate(model, data, args)
        val_acc_history.append(test_acc)
        loss_history.append(test_loss.item() if hasattr(test_loss, 'item') else test_loss)  # Ensure storing numerical values
        if test_acc > best_acc:
            patience = 0
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(),'model.pth')
            if H is not None:
                args.num_edges = H[0].shape[1]
        else:
            patience = patience + 1

        if patience > args.patience:
            break

        # Only print detailed info in first few and last few epochs
        if epoch < 5 or epoch % 100 == 0 or epoch > args.epoch - 5:
            print("========================> ", epoch)
            print("Train acc: {}, loss: {}".format(acc,loss))
            # print("Val acc: {}, loss: {}".format(val_acc,val_loss))
            print("Test acc: {}, loss: {}".format(test_acc,test_loss))
            # print("Epoch: {}; Loss: {}, acc: {}".format(i,loss,acc))
            print("Time: {} (h:mm:ss)".format(format_time(time.time()-t0)))
        
        spent_time.append(format_time(time.time()-t0))
        

    print("Best epch: {}, acc: {}".format(best_epoch, best_acc))
    return best_acc, val_acc_history, loss_history

def evaluate(model, data, args):
    stage = args.stage

    model.eval()

    # Select correct mask based on stage
    if stage == 'test':
        mask = data['test_idx']
    else:
        mask = data['val_idx']
    labels = data['lbls'][mask]

    out, x, H, H_raw = model(data,args)
    # pred = F.log_softmax(out, dim=1)
    labels = labels.float()
    contra_ls = 0
    contra_ls2 = 0
    if H_raw is not None:
        contra_ls = contrast_loss(H_raw,mask,labels)
        contra_ls2 = contrast_loss2(H, x)
    loss_function = nn.BCEWithLogitsLoss()
    loss = loss_function(out[mask], labels) + contra_ls * args.namuda + contra_ls2 * (args.namuda2 / 1000)

    # _, pred = pred[mask].max(dim=1)
    # correct = int(pred.eq(labels).sum().item())
    # acc = correct / len(labels)
    outs, lbls = out[mask], labels
    lbls_ = lbls.cpu()
    outputs = outs.cpu()
    preds = np.where(F.sigmoid(outputs) > 0.5, 1, 0)
    TP = sum((lbls_ == 1) & (preds == 1))
    FP = sum((lbls_ == 0) & (preds == 1))
    FN = sum((lbls_ == 1) & (preds == 0))
    TN = sum((lbls_ == 0) & (preds == 0))
    TP = TP.sum().item()
    FP = FP.sum().item()
    FN = FN.sum().item()
    TN = TN.sum().item()
    # Calculate accuracy based on above values
    acc = (TP + TN) / (TP + FP + FN + TN)
    return acc, loss

def train_dhl(data, args):
    in_dim = args.in_dim
    hid_dim = args.hid_dim 
    out_dim = args.out_dim
    num_edges = args.num_edges
    model = HGNN_classifier(args)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate,weight_decay=args.wdecay)

    best_acc, val_acc_history, loss_history = train(model, optimizer, data, args)
    
    return best_acc, val_acc_history, loss_history

def train_gcn(data, args):
    in_dim = args.in_dim
    hid_dim = args.hid_dim 
    out_dim = args.out_dim
    model = GCN(args)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate,weight_decay=args.wdecay)

    best_acc, val_acc_history, loss_history = train(model, optimizer, data, args)

    return best_acc, val_acc_history, loss_history

def train_gat(data, args):
    model = GAT(args)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate,weight_decay=args.wdecay)

    best_acc, val_acc_history, loss_history = train(model, optimizer, data, args)

    return best_acc, val_acc_history, loss_history

def train_mlp(data, args):
    model = MLP(args)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lrate,weight_decay=args.wdecay)
    best_acc, val_acc_history, loss_history = train(model, optimizer, data, args)

    return best_acc, val_acc_history, loss_history