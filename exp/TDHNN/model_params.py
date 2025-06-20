import os
# Set environment variables, must be before importing numpy and torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import argparse
from networks import HGNN_classifier, GCN, GAT, MLP
import numpy as np

# Create argument parser, simulating args in main program
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dim', type=int, default=1024)  # Input feature dimension
    parser.add_argument('--hid_dim', type=int, default=64)   # Hidden layer dimension
    parser.add_argument('--out_dim', type=int, default=5)    # Output class number
    parser.add_argument('--num_edges', type=int, default=15) # Initial number of hyperedges
    parser.add_argument('--min_num_edges', type=int, default=32) # Minimum number of hyperedges
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--conv_number', type=int, default=1) # Number of convolution layers
    parser.add_argument('--k_n', type=int, default=10)  # Number of selected nodes
    parser.add_argument('--k_e', type=int, default=10)  # Number of selected edges
    parser.add_argument('--backbone', type=str, default='linear')
    parser.add_argument('--transfer', type=int, default=1)
    args = parser.parse_args([])  # Empty list means not reading parameters from command line
    return args

# Initialize parameters
args = create_args()

# Force using CPU to avoid CUDA-related errors
device = torch.device('cpu')
args.device = 'cpu'

# Initialize all models
models = {
    "TDHNN": HGNN_classifier(args),
    "GCN": GCN(args),
    "GAT": GAT(args),
    "MLP": MLP(args)
}

# Calculate and print parameters for each model
print("\nModel Parameter Comparison:")
for name, model in models.items():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{name} Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print parameter count for each component
    if name == "TDHNN":
        print("\nParameter count for each component:")
        
        # Linear backbone
        linear_backbone_params = sum(p.numel() for n, p in model.named_parameters() if "linear_backbone" in n)
        print(f"Linear backbone: {linear_backbone_params:,}")
        
        # Convolution layers
        conv_params = sum(p.numel() for n, p in model.named_parameters() if "convs" in n)
        print(f"Hypergraph convolution layers: {conv_params:,}")
        
        # Hypergraph constructor
        constructor_params = sum(p.numel() for n, p in model.named_parameters() if "HConstructor" in n)
        print(f"Hypergraph constructor: {constructor_params:,}")
        
        # Classifier
        classifier_params = sum(p.numel() for n, p in model.named_parameters() if "classifier" in n)
        print(f"Classifier: {classifier_params:,}")

    # Optional: print model structure
    print(f"Model structure:")
    print(model) 