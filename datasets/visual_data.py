from PROSE_HGNN.datasets.data_helper import load_ft
from PROSE_HGNN.datasets.data_helper import load
from PROSE_HGNN.HGNN_utils import hypergraph_utils as hgut
import numpy as np
import torch
from PROSE_HGNN.graph_learners import *
import pandas as pd

def load_feature_construct_H(X,
                             Y,
                             idx,
                             m_prob=1,
                             K_neigs=[10],
                             is_probH=True,
                             split_diff_scale=False,):
    ft, lbls, idx_train, idx_test, idx_val = load(X , Y, idx)
    fts = None

    fts = hgut.feature_concat(fts, ft)
    if fts is None:
        raise Exception(f'None feature used for model!')
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    tmp = hgut.construct_H_with_KNN(ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)  # (12311, 12311)
    H = hgut.hyperedge_concat(H, tmp)
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')
    adj_original = H
    features = fts
    labels = lbls
    idx_train_all = np.zeros(len(idx))
    idx_test_all = np.zeros(len(idx))
    idx_val_all = np.zeros(len(idx))
    idx_train_all[idx_train] = 1
    idx_test_all[idx_test] = 1
    idx_val_all[idx_val] = 1
    train_mask = idx_train_all
    test_mask = idx_test_all
    val_mask = idx_val_all
    nclasses = int(lbls.shape[1])
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    return features,  labels, nclasses, train_mask, test_mask, val_mask, adj_original
