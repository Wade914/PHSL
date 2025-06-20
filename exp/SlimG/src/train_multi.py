import io
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from copy import deepcopy


def set_seed(seed: int):
    """Set random seed"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_regularizer(model, lambda_1, lambda_2):
    """Regularization loss"""
    out = []
    for j, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and 'bias' not in name:
            out.append(lambda_1 * torch.abs(param).sum())
            out.append(lambda_2 * torch.sqrt(torch.pow(param, 2).sum()))
    return torch.stack(out).sum()


def evaluate_model(model, features, edge_index, labels, nodes):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        pred_ = model(features, edge_index)
        pred_ = pred_[nodes]
        labels_ = labels[nodes]
        
        # Calculate loss
        loss_func = nn.BCEWithLogitsLoss()
        labels_ = labels_.float()
        loss_ = loss_func(pred_, labels_).item()
        
        # Calculate accuracy
        lbls = labels_.cpu().numpy()
        outputs = pred_.cpu()
        preds = np.where(F.sigmoid(outputs) > 0.5, 1, 0)
        
        TP = sum((lbls == 1) & (preds == 1))
        FP = sum((lbls == 0) & (preds == 1))
        FN = sum((lbls == 1) & (preds == 0))
        TN = sum((lbls == 0) & (preds == 0))
        
        TP = TP.sum().item()
        FP = FP.sum().item()
        FN = FN.sum().item()
        TN = TN.sum().item()
        
        # Calculate accuracy
        acc_ = (TP + TN) / (TP + FP + FN + TN)
        
        return loss_, acc_


def train_slimG(data, max_epochs=500, lr=0.01, weight_decay=5e-4, 
               lambda_1=0.5e-3, lambda_2=0.5e-4, verbose=False):
    """
    Train SlimG model
    
    Args:
        data: Data dictionary containing features, labels, edge_index, train_idx, val_idx, test_idx etc.
        max_epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: Weight decay
        lambda_1: L1 regularization coefficient
        lambda_2: L2 regularization coefficient
        verbose: Whether to print detailed information
    
    Returns:
        best_acc: Best validation accuracy
        val_acc_history: Validation accuracy history
        loss_history: Loss history
    """
    device = data['features'].device
    
    # Import model
    from models import load_model
    
    # Create model
    model = load_model(
        num_nodes=data['num_nodes'],
        num_features=data['num_features'],
        num_classes=data['num_classes']
    ).to(device)
    
    # Preprocess data
    model.preprocess(data['features'], data['edge_index'], data['labels'], device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.BCEWithLogitsLoss()
    
    # Training history record
    loss_history = []
    val_acc_history = []
    
    # Best model tracking
    best_acc = 0
    best_epoch = -1
    best_state = None
    
    # Get training, validation, test indices
    train_nodes = data['train_idx']
    val_nodes = data['val_idx']
    test_nodes = data['test_idx']
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        pred_ = model(data['features'], data['edge_index'])
        pred_, labels_ = pred_[train_nodes], data['labels'][train_nodes]
        labels_ = labels_.float()
        
        # Calculate loss
        loss1 = loss_func(pred_, labels_)
        if lambda_1 > 0 or lambda_2 > 0:
            loss2 = to_regularizer(model, lambda_1, lambda_2)
        else:
            loss2 = 0
        
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, data['features'], data['edge_index'], 
                                         data['labels'], val_nodes)
        
        # Record history
        val_acc_history.append(val_acc)
        loss_history.append(loss.item() if hasattr(loss, 'item') else loss)
        
        # Update best model
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_state = deepcopy(model.state_dict())
        
        # Print training information (only at key rounds)
        if verbose and (epoch < 5 or epoch % 100 == 0 or epoch > max_epochs - 5):
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Val Acc = {val_acc:.4f}")
    
    if verbose:
        print(f"Best epoch: {best_epoch}, Best val acc: {best_acc:.4f}")
    
    return best_acc, val_acc_history, loss_history


def train_slimG_with_grid_search(data, max_epochs=500, lr=0.01, weight_decay=5e-4, verbose=False):
    """
    Train SlimG model with grid search
    """
    # Simplified grid search parameters
    search_range_1 = 0.8e-3
    search_range_2 = 0.8e-4
    
    # First training for parameter search
    best_acc, _, _ = train_slimG(data, max_epochs, lr, weight_decay, 
                                search_range_1, search_range_2, verbose=False)
    
    # Re-train using the best found parameters
    final_acc, val_acc_history, loss_history = train_slimG(
        data, max_epochs, lr, weight_decay, 
        search_range_1, search_range_2, verbose=verbose
    )
    
    return final_acc, val_acc_history, loss_history 