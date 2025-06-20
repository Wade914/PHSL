# import dgl.function as fn
import torch
from models import HGNN_conv
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
EOS = 1e-10
import HGNN_utils.hypergraph_utils as hgut

class HGNNConv_dense(nn.Module):


    def __init__(self, in_features,hid_features, out_features, bias=False, batch_norm=False):
        self.dropout = 0.5
        super(HGNNConv_dense, self).__init__()
        # self.weight = torch.Tensor(in_features, out_features)
        # print('weight',self.weight.shape)
        # self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        # print('infeature',in_features)
        self.hgc1 = HGNN_conv(in_features, hid_features)  # HGNN中全连接参数
        self.hgc2 = HGNN_conv(hid_features, out_features)
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        adj = torch.tensor(adj)
        input = input.double()
        # support = torch.matmul(input, self.weight.double())
        G = hgut.generate_G_from_H(adj).to(adj.device)
        # a = torch.matmul(adj.double(), adj.T.double()).double()
        # a = torch.matmul(adj.T().double(), support.double()).double()
        # support = torch.matmul(input, self.weight)
        # output = torch.matmul(adj, support)
        # output = torch.matmul(G, support.double())
        x = F.relu(self.hgc1(input, G))  # 一层HGNN
        x = F.dropout(x, self.dropout)
        output = self.hgc2(x, G)
        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())
