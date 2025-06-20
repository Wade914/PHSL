import os
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
from utils import setup_seed, arg_parse
from load_data import load_data
from train import train_dhl
from networks import HGNN_classifier

def plot_loss_with_acc(loss_history, val_acc_history):
    # Comment out graphical display part, keep only data processing logic
    # Handle possible tensor format
    if len(loss_history) > 0:
        if hasattr(loss_history[-1], 'item'):
            final_loss = loss_history[-1].item()
        else:
            final_loss = loss_history[-1]
    else:
        final_loss = 0.0
        
    if len(val_acc_history) > 0:
        best_val_acc = max(val_acc_history)
    else:
        best_val_acc = 0.0
        
    print(f"Training completed. Final loss: {final_loss:.4f}, Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    print("=" * 100)
    print("Starting multi-dataset TDHNN model experiment - TDHNN model runs 5 independent experiments on each dataset")
    print("=" * 80)
    
    # Define dataset suffixes to test
    dataset_suffixes = [-5, -10, -20, -30, -40, -50]
    
    # Only test TDHNN model
    models_to_test = [
        ("TDHNN", "dhl")
    ]
    
    # Corresponding training function mapping
    chosse_trainer = {
        'dhl': train_dhl
    }
    
    # Corresponding model mapping
    chosse_model = {
        'dhl': HGNN_classifier
    }
    
    # Store all datasets and model results
    all_datasets_results = {}
    
    # Create result save file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"TDHNN_experiment_results_{timestamp}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("TDHNN model multi-dataset experiment results\n")
        f.write(f"Experiment time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
    
    for suffix in dataset_suffixes:
        print(f"\n{'#'*50} Dataset suffix: {suffix} {'#'*50}")
        
        # Store results for current dataset
        dataset_results = {}
        
        # Record dataset information
        dataset_info = None
        
        for model_name, model_key in models_to_test:
            print(f"\n{'='*25} Dataset_{suffix} - Model: {model_name} {'='*25}")
            
            test_acc_history = []
            model_parameters_info = []
            
            for trial in range(5):
                print(f"\n{'-'*15} Dataset_{suffix} - {model_name} - Experiment {trial+1}/5 {'-'*15}")
                
                # Set random seed
                setup_seed(trial)
                
                # Get parameters
                args = arg_parse()
                args.model = model_key
                args.dataset = '40'  # Set default dataset type
                
                try:
                    # Load data
                    data = load_data(args, suffix)
                    
                    fts = data['fts']
                    lbls = data['lbls']
                    
                    args.in_dim = fts.shape[1]
                    args.out_dim = lbls.shape[1]  # For multi-label classification, use the number of columns in labels
                    args.min_num_edges = args.k_e
                    
                    # Only record dataset information for the first model's first experiment
                    if model_name == models_to_test[0][0] and trial == 0:
                        dataset_info = {
                            'suffix': suffix,
                            'features_shape': fts.shape,
                            'labels_shape': lbls.shape,
                            'num_classes': args.out_dim,
                            'train_size': len(data['train_idx']),
                            'val_size': len(data['val_idx']),
                            'test_size': len(data['test_idx'])
                        }
                        
                        print(f"Dataset_{suffix} information:")
                        print(f"Feature matrix X shape: {fts.shape}")
                        print(f"Label matrix shape: {lbls.shape}, Number of classes: {args.out_dim}")
                        print(f"Training set sample count: {len(data['train_idx'])}")
                        print(f"Validation set sample count: {len(data['val_idx'])}")
                        print(f"Test set sample count: {len(data['test_idx'])}")
                        print(f"Using device: {args.device}")
                
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    print(f"Skipping Dataset_{suffix}")
                    break
                except Exception as e:
                    print(f"Error loading data: {e}")
                    print(f"Skipping Dataset_{suffix}")
                    break
                
                # Train model
                print(f"Starting training {model_name}...")
                
                try:
                    # Call corresponding training function
                    best_acc, val_acc_history, loss_history = chosse_trainer[model_key](data, args)
                    
                    # Only print model information for the first experiment of each model
                    if trial == 0:
                        model = chosse_model[model_key](args)
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print(f"{model_name} Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
                        
                        # Store model parameter information
                        model_info = {
                            "model": model_name,
                            "parameters": total_params,
                            "accuracy": best_acc
                        }
                        model_parameters_info.append(model_info)
                    
                    print(f"Dataset_{suffix} - {model_name} Training completed! Best test accuracy: {best_acc:.5f}")
                    
                    # Plot loss and accuracy curve (no graphical display)
                    plot_loss_with_acc(loss_history, val_acc_history)
                    
                    print(f"Dataset_{suffix} - {model_name} Experiment{trial+1} Final test accuracy: {best_acc:.5f}")
                    test_acc_history.append(best_acc)
                    
                except Exception as e:
                    print(f"Error during training: {e}")
                    print(f"Skipping current experiment")
                    continue
            
            # If experiment completed successfully, calculate statistical results
            if test_acc_history:
                # Calculate statistical results for current model
                mean_acc = np.mean(test_acc_history)
                std_acc = np.std(test_acc_history)
                
                print(f"\nDataset_{suffix} - {model_name} Model results statistics:")
                print(f"All experiment results: {[f'{acc:.4f}' for acc in test_acc_history]}")
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
        
        # If current dataset has results, perform summary
        if dataset_results:
            all_datasets_results[suffix] = {
                'dataset_info': dataset_info,
                'model_results': dataset_results
            }
            
            # Display TDHNN model results for current dataset
            print(f"\n{'='*30} Dataset_{suffix} TDHNN model results {'='*30}")
            
            for model_name, results in dataset_results.items():
                print(f"TDHNN model statistics:")
                print(f"Average accuracy: {results['mean']:.4f} ± {results['std']:.4f}")
                print(f"Highest accuracy: {results['max']:.4f}")
                print(f"Parameter count: {results['parameters']:,}")
            
            # Save to file
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nDataset_{suffix} Results:\n")
                f.write("=" * 50 + "\n")
                if dataset_info:
                    f.write(f"Dataset information: Feature shape{dataset_info['features_shape']}, Number of classes{dataset_info['num_classes']}, ")
                    f.write(f"Training set{dataset_info['train_size']}, Validation set{dataset_info['val_size']}, Test set{dataset_info['test_size']}\n")
                
                for model_name, results in dataset_results.items():
                    f.write(f"TDHNN model results:\n")
                    f.write(f"Average accuracy: {results['mean']:.4f} ± {results['std']:.4f}\n")
                    f.write(f"Highest accuracy: {results['max']:.4f}\n")
                    f.write(f"Parameter count: {results['parameters']:,}\n")
                    f.write(f"All experiments: {results['accuracy_list']}\n")
                f.write("\n" + "="*80 + "\n\n")
    
    # Final summary of all results
    print("\n" + "#" * 100)
    print("Final comparison results of TDHNN model on all datasets")
    print("#" * 100)
    
    # Create summary table
    summary_results = []
    for suffix, data in all_datasets_results.items():
        for model_name, results in data['model_results'].items():
            summary_results.append({
                'dataset': f"Dataset_{suffix}",
                'mean_acc': results['mean'],
                'std_acc': results['std'],
                'max_acc': results['max'],
                'parameters': results['parameters']
            })
    
    # Sort by average accuracy
    summary_results.sort(key=lambda x: x['mean_acc'], reverse=True)
    
    print(f"{'Dataset':<15} {'Mean Accuracy':<12} {'Std Dev':<8} {'Max Accuracy':<12} {'Parameters':<12}")
    print("-" * 70)
    for result in summary_results:
        print(f"{result['dataset']:<15} {result['mean_acc']:.4f}       {result['std_acc']:.4f}   {result['max_acc']:.4f}       {result['parameters']:,}")
    
    # Save final summary results
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "#" * 100 + "\n")
        f.write("Final Summary Results - TDHNN Performance on All Datasets (Sorted by Average Accuracy)\n")
        f.write("#" * 100 + "\n")
        f.write(f"{'Dataset':<15} {'Mean Accuracy':<12} {'Std Dev':<8} {'Max Accuracy':<12} {'Parameters':<12}\n")
        f.write("-" * 70 + "\n")
        for result in summary_results:
            f.write(f"{result['dataset']:<15} {result['mean_acc']:.4f}       {result['std_acc']:.4f}   {result['max_acc']:.4f}       {result['parameters']:,}\n")
        
        f.write(f"\nExperiment completion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Find best dataset result
    if summary_results:
        best_overall = summary_results[0]
        print(f"\nTDHNN Best Performance: {best_overall['dataset']} (Average accuracy: {best_overall['mean_acc']:.4f})")
    
    print(f"\nAll experiment results saved to file: {results_file}")
    print("Experiment completed!") 