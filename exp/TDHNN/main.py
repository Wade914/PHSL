from utils import setup_seed, arg_parse, visualization
from load_data import load_data
from train import train_dhl,train_gcn,train_mlp,train_gat
import json
import torch
print(torch.__version__)

from networks import HGNN_classifier, GCN, GAT, MLP
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
def plot_loss_with_acc(loss_history, val_acc_history):
    loss_history = torch.tensor(loss_history).cpu()
    val_acc_history = torch.tensor(val_acc_history).cpu()
    loss_history = loss_history.tolist()
    val_acc_history = val_acc_history.tolist()
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c='blue')
    # plt.ylabel('Loss')
    ax1.set_ylabel('Loss', color='blue', fontsize=14)  # Set y-axis color to orange
    ax1.tick_params(axis='y', colors='blue', labelsize=12)  # Set y-axis tick color to orange

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c='red')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    # plt.ylabel('ValAcc')
    ax2.set_ylabel('Accuracy', color='red', fontsize=14)  # Set y-axis color to blue
    ax2.tick_params(axis='y', colors='red', labelsize=12)  # Set y-axis tick color to blue
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

chosse_trainer = {
    'dhl':train_dhl,
    'gcn':train_gcn,
    'MLP':train_mlp,
    'gat':train_gat
}

args = arg_parse()

setup_seed(args.seed)
data = load_data(args)

fts = data['fts']
lbls = data['lbls']

args.in_dim = fts.shape[1]
args.out_dim = lbls.shape[1]
args.min_num_edges = args.k_e

args_list = []

best_acc, val_acc_history, loss_history= chosse_trainer[args.model](data, args)

args.best_acc = best_acc
args_list.append(args.__dict__)

############################################## visualization
chosse_model = {
    'dhl':HGNN_classifier,
    'gcn':GCN,
    'MLP':MLP,
    'gat':GAT
}

model = chosse_model[args.model](args)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{args.model.upper()} Model Summary:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
state_dict = torch.load('model.pth',map_location=args.device)
model.load_state_dict(state_dict)
model.to(args.device)

# Calculate and print model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTDHNN Model Summary:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Print component parameters
linear_backbone_params = sum(p.numel() for n, p in model.named_parameters() if "linear_backbone" in n)
conv_params = sum(p.numel() for n, p in model.named_parameters() if "convs" in n)
constructor_params = sum(p.numel() for n, p in model.named_parameters() if "HConstructor" in n)
classifier_params = sum(p.numel() for n, p in model.named_parameters() if "classifier" in n)

print("\nParameter count for each component:")
print(f"Linear backbone: {linear_backbone_params:,}")
print(f"Hypergraph convolution layers: {conv_params:,}")
print(f"Hypergraph constructor: {constructor_params:,}")
print(f"Classifier: {classifier_params:,}")

model.eval()
mask = data['test_idx']
labels = data['lbls'][mask]

out, x, H, H_raw = model(data,args)
# pred = F.log_softmax(out, dim=1)

# _, pred = pred[mask].max(dim=1)
# correct = int(pred.eq(labels).sum().item())
# acc = correct / len(labels)
outs, lbls = out[mask], lbls[mask]
lbls_ = lbls.cpu().numpy()
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
print("Acc ===============> ", acc)
plot_loss_with_acc(loss_history, val_acc_history)
# visualization(model, data, args, title=None)
# with open('commandline_args{}.txt'.format(args.cuda), 'w') as f:
#     json.dump([args.__dict__,args.__dict__], f, indent=2)