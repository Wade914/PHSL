import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from copy import deepcopy
import pandas as pd
import pickle
import torch
import random
import numpy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.models import HGNN, HGNNP, HNHN, HyperGCN, GCN, GAT, GIN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import numpy as np
import dhg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from torchsummary import summary

def save_as_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def plot_loss_with_acc(loss_history, val_acc_history):
    # Commented out graph display parts, only keeping data processing logic
    print(f"Training completed. Final loss: {loss_history[-1]:.4f}, Best val acc: {max(val_acc_history):.4f}")
    # fig = plt.figure(figsize=(8, 6))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(range(len(loss_history)), loss_history,
    #          c='blue')
    # # plt.ylabel('Loss')
    # ax1.set_ylabel('Loss', color='blue', fontsize=14)
    # ax1.tick_params(axis='y', colors='blue', labelsize=12)

    # ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    # ax2.plot(range(len(val_acc_history)), val_acc_history,
    #          c='red')
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")
    # # plt.ylabel('ValAcc')
    # ax2.set_ylabel('Accuracy', color='red', fontsize=14)
    # ax2.tick_params(axis='y', colors='red', labelsize=12)
    # ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    # plt.xlabel('Epoch')
    # plt.title('Training Loss & Validation Accuracy')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()

def train(net, X, G, lbls, train_idx, optimizer, epoch, loss_history):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    lbls = lbls.float()
    loss_function = nn.BCEWithLogitsLoss()
    lbls = lbls.to(device)
    loss = loss_function(outs, lbls)
    # loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    # Only print detailed information for first few epochs and last few epochs
    if epoch < 5 or epoch % 100 == 0 or epoch > 495:
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    loss_history.append(loss.item())
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    lbls = lbls.to(device)
    lbls_ = lbls.cpu().numpy()
    outputs = outs.cpu()
    preds = np.where(F.sigmoid(outputs) > 0.5, 1, 0)
    # lbls_ = lbls.long()

    if not test:
        # res = evaluator.validate(lbls_, outs)
        TP = sum((lbls_ == 1) & (preds == 1))
        FP = sum((lbls_ == 0) & (preds == 1))
        FN = sum((lbls_ == 1) & (preds == 0))
        TN = sum((lbls_ == 0) & (preds == 0))
        TP = TP.sum().item()
        FP = FP.sum().item()
        FN = FN.sum().item()
        TN = TN.sum().item()
        # print(TP + FP + FN + TN)
        # Calculate Accuracy based on the values above
        accuarcy = (TP + TN) / (TP + FP + FN + TN)
    else:
        # res = evaluator.test(lbls_, outs)
        TP = sum((lbls_ == 1) & (preds == 1))
        FP = sum((lbls_ == 0) & (preds == 1))
        FN = sum((lbls_ == 1) & (preds == 0))
        TN = sum((lbls_ == 0) & (preds == 0))
        TP = TP.sum().item()
        FP = FP.sum().item()
        FN = FN.sum().item()
        TN = TN.sum().item()
        # print(TP + FP + FN + TN)
        # Calculate Accuracy based on the values above
        accuarcy = (TP + TN) / (TP + FP + FN + TN)
    return accuarcy


if __name__ == "__main__":
    print("=" * 100)
    print("Starting multi-dataset, multi-GNN model comparison experiment - each model runs 5 independent experiments on each dataset")
    print("=" * 100)
    
    # Define dataset suffixes to test
    dataset_suffixes = [-20, -30, -40, -50]
    
    # Define GNN model list to test
    models_to_test = [
        # ("GCN", lambda n_features, class_num: GCN(n_features, 64, class_num)),
        # ("GAT", lambda n_features, class_num: GAT(n_features, 64, class_num, 4, use_bn=True)),
        ("GIN", lambda n_features, class_num: GIN(n_features, 64, class_num, 3))
    ]
    
    # Store results for all datasets and models
    all_datasets_results = {}
    
    # Create results save file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"GNN_experiment_results_{timestamp}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("GNN Model Multi-Dataset Comparison Experiment Results\n")
        f.write(f"Experiment Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
    
    for suffix in dataset_suffixes:
        print(f"\n{'#'*50} Dataset suffix: {suffix} {'#'*50}")
        
        # Store results for all models on current dataset
        dataset_results = {}
        
        # Record dataset information
        dataset_info = None
        
        for model_name, model_constructor in models_to_test:
            print(f"\n{'='*25} Dataset_{suffix} - Model: {model_name} {'='*25}")
            
            test_acc_history = []
            model_parameters_info = []
            
            for trial in range(5):
                print(f"\n{'-'*15} Dataset_{suffix} - {model_name} - Experiment {trial+1}/5 {'-'*15}")
                loss_history = []
                val_acc_history = []
                test_results = []
                set_seed(trial)  # Use fixed seed to ensure reproducibility
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
            # ********************************************
                try:
                    X = pd.read_csv(f'/root/autodl-tmp/exp/data/X_{suffix}.csv', header=None, skiprows=1)
                    X = np.array(X)
                    lbl = pd.read_csv(f'/root/autodl-tmp/exp/data/Y_{suffix}.csv', header=None, skiprows=1)
                    lbl = np.array(lbl)
                    class_num = int(lbl.shape[1])
                    idx = pd.read_csv(f'/root/autodl-tmp/exp/data/idx_{suffix}.csv', header=None, skiprows=1)
                    idx = np.array(idx)
                    
                    edge_list = dhg.datapipe.load_from_pickle(f'/root/autodl-tmp/exp/data/H_{suffix}.pkl')
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    print(f"Skipping dataset_{suffix}")
                    break
                
                num_vertices = X.shape[0]
                
                # Only record dataset information during first experiment of first model
                if model_name == models_to_test[0][0] and trial == 0:
                    dataset_info = {
                        'suffix': suffix,
                        'features_shape': X.shape,
                        'labels_shape': lbl.shape,
                        'num_classes': class_num,
                        'num_edges': len(edge_list),
                        'num_vertices': num_vertices,
                        'max_node_index': np.array([max(edge) for edge in edge_list if edge]).max() if edge_list else 0
                    }
                    
                    print(f"Dataset_{suffix} information:")
                    print(f"Feature matrix X shape: {X.shape}")
                    print(f"Label matrix shape: {lbl.shape}, number of classes: {class_num}")
                    print(f"Hypergraph edges: {len(edge_list)}")
                    print(f"Max node index: {dataset_info['max_node_index']}")
                    print(f"Total nodes: {num_vertices}")
                    print(f"Using device: {device}")
                
                # Build hypergraph and regular graph
                HG = Hypergraph(num_vertices, edge_list)
                G = Graph.from_hypergraph_clique(HG, weighted=True)
                
                # Save edge information (only during first run)
                if model_name == models_to_test[0][0] and trial == 0:
                    edge_A = G.e_dst
                    edge_B = G.e_src
                    save_as_pickle(edge_A, f'/root/autodl-tmp/exp/data/edge_A1_{suffix}.pkl')
                    save_as_pickle(edge_B, f'/root/autodl-tmp/exp/data/edge_B1_{suffix}.pkl')
                
                train_mask = np.where(idx == 1)[0]  # Training set indices
                test_mask = np.where(idx == 2)[0]
                val_mask = np.where(idx == 0)[0]
                n_features = X.shape[1]
                
                # Only print dataset split information during first experiment of first model
                if model_name == models_to_test[0][0] and trial == 0:
                    print(f"Training samples: {len(train_mask)}, validation samples: {len(val_mask)}, test samples: {len(test_mask)}")
            # ********************************************
                # Create corresponding model
                net = model_constructor(n_features, class_num)
                optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
            # ********************************************
                X = torch.tensor(X).float()
                lbl = torch.tensor(lbl)
                X, lbl = X.to(device), lbl.to(X.device)
                G = G.to(X.device)
                net = net.to(X.device)
                
                # Only print model information during first experiment of each model
                if trial == 0:
                    total_params = sum(p.numel() for p in net.parameters())
                    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    print(f"{model_name} model parameters: total={total_params:,}, trainable={trainable_params:,}")
                
                best_state = None
                best_epoch, best_val = 0, 0
                
                print(f"Starting training {model_name}...")
                for epoch in range(500):
                    # train
                    train(net, X, G, lbl, train_mask, optimizer, epoch, loss_history)
                    # validation
                    if epoch % 1 == 0:
                        with torch.no_grad():
                            val_res = infer(net, X, G, lbl, val_mask)
                        val_acc_history.append(val_res)
                        if val_res > best_val:
                            # Only print when validation accuracy improves significantly
                            if val_res - best_val > 0.001:
                                print(f"Validation accuracy improved: {best_val:.5f} -> {val_res:.5f} (epoch {epoch})")
                            best_epoch = epoch
                            best_val = val_res
                            best_state = deepcopy(net.state_dict())
                
                print(f"Dataset_{suffix} - {model_name} training completed! Best validation accuracy: {best_val:.5f} (epoch {best_epoch})")
                
                # test
                plot_loss_with_acc(loss_history, val_acc_history)
                net.load_state_dict(best_state)
                res = infer(net, X, G, lbl, test_mask, test=True)
                print(f"Dataset_{suffix} - {model_name} experiment{trial+1} final test accuracy: {res:.5f}")
                test_acc_history.append(res)
                
                # Store model parameter information
                if trial == 0:  # Only store during first experiment
                    model_info = {
                        "model": model_name,
                        "parameters": sum(p.numel() for p in net.parameters()),
                        "accuracy": res
                    }
                    model_parameters_info.append(model_info)
            
            # If experiments completed successfully, calculate statistical results
            if test_acc_history:
                # Calculate statistical results for current model
                mean_acc = np.mean(test_acc_history)
                std_acc = np.std(test_acc_history)
                
                print(f"\nDataset_{suffix} - {model_name} model result statistics:")
                print(f"Individual experiment results: {[f'{acc:.4f}' for acc in test_acc_history]}")
                print(f"Average test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"Highest accuracy: {max(test_acc_history):.4f}")
                print(f"Lowest accuracy: {min(test_acc_history):.4f}")
                
                # Store results
                dataset_results[model_name] = {
                    'accuracy_list': test_acc_history,
                    'mean': mean_acc,
                    'std': std_acc,
                    'max': max(test_acc_history),
                    'min': min(test_acc_history),
                    'parameters': model_parameters_info[0]['parameters'] if model_parameters_info else 0
                }
        
        # If current dataset has results, summarize
        if dataset_results:
            all_datasets_results[suffix] = {
                'dataset_info': dataset_info,
                'model_results': dataset_results
            }
            
            # Display model comparison for current dataset
            print(f"\n{'='*30} Dataset_{suffix} Model Comparison Results {'='*30}")
            print(f"{'Model Name':<12} {'Average Accuracy':<15} {'Std Dev':<10} {'Highest Accuracy':<12} {'Parameters':<15}")
            print("-" * 80)
            
            # Sort by average accuracy
            sorted_results = sorted(dataset_results.items(), key=lambda x: x[1]['mean'], reverse=True)
            
            for model_name, results in sorted_results:
                print(f"{model_name:<12} {results['mean']:.4f}          {results['std']:.4f}     {results['max']:.4f}       {results['parameters']:,}")
            
            # Save to file
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nDataset_{suffix} results:\n")
                f.write("=" * 50 + "\n")
                if dataset_info:
                    f.write(f"Dataset info: feature shape{dataset_info['features_shape']}, classes{dataset_info['num_classes']}, ")
                    f.write(f"edges{dataset_info['num_edges']}, nodes{dataset_info['num_vertices']}\n")
                f.write(f"{'Model Name':<12} {'Average Accuracy':<15} {'Std Dev':<10} {'Highest Accuracy':<12} {'Parameters':<15}\n")
                f.write("-" * 80 + "\n")
                for model_name, results in sorted_results:
                    f.write(f"{model_name:<12} {results['mean']:.4f}          {results['std']:.4f}     {results['max']:.4f}       {results['parameters']:,}\n")
                f.write("\nDetailed results:\n")
                for model_name, results in sorted_results:
                    f.write(f"{model_name}: {results['mean']:.4f} ± {results['std']:.4f}\n")
                    f.write(f"  Individual experiments: {results['accuracy_list']}\n")
                f.write("\n" + "="*80 + "\n\n")
    
    # Final summary of all results
    print("\n" + "#" * 100)
    print("Final comparison results for all datasets and models")
    print("#" * 100)
    
    # Create summary table
    summary_results = []
    for suffix, data in all_datasets_results.items():
        for model_name, results in data['model_results'].items():
            summary_results.append({
                'dataset': f"Dataset_{suffix}",
                'model': model_name,
                'mean_acc': results['mean'],
                'std_acc': results['std'],
                'max_acc': results['max'],
                'parameters': results['parameters']
            })
    
    # Sort by average accuracy
    summary_results.sort(key=lambda x: x['mean_acc'], reverse=True)
    
    print(f"{'Dataset':<15} {'Model':<8} {'Average Accuracy':<12} {'Std Dev':<8} {'Highest Accuracy':<12} {'Parameters':<12}")
    print("-" * 85)
    for result in summary_results:
        print(f"{result['dataset']:<15} {result['model']:<8} {result['mean_acc']:.4f}       {result['std_acc']:.4f}   {result['max_acc']:.4f}       {result['parameters']:,}")
    
    # Save final summary results
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "#" * 100 + "\n")
        f.write("Final Summary Results - Sorted by Average Accuracy\n")
        f.write("#" * 100 + "\n")
        f.write(f"{'Dataset':<15} {'Model':<8} {'Average Accuracy':<12} {'Std Dev':<8} {'Highest Accuracy':<12} {'Parameters':<12}\n")
        f.write("-" * 85 + "\n")
        for result in summary_results:
            f.write(f"{result['dataset']:<15} {result['model']:<8} {result['mean_acc']:.4f}       {result['std_acc']:.4f}   {result['max_acc']:.4f}       {result['parameters']:,}\n")
        
        f.write(f"\nExperiment completion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Find overall best result
    best_overall = summary_results[0]
    print(f"\nOverall best result: {best_overall['dataset']} - {best_overall['model']} (average accuracy: {best_overall['mean_acc']:.4f})")
    
    print(f"\nAll experiment results saved to file: {results_file}")
    print("Experiment completed!")