import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, use_bn=True):
        super(HGNN_conv, self).__init__()
        self.bn = nn.BatchNorm1d(out_ft) if use_bn else None
        # print(in_ft,out_ft)
        self.weight = Parameter(torch.Tensor(out_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.theta = nn.Linear(in_ft, out_ft, bias=bias)
        self.drop = nn.Dropout(0.5)
        # self.weight = self.weight.data.to(dtype)
        # if self.bias is not None:
        #     self.bias = self.bias.data.to(dtype)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):

        x = x.to(torch.float32)
        x = self.theta(x)
        x = self.drop(x)
        # x = x.matmul(self.weight)
        # x = x.double()
        G = G.float()
        if self.bias is not None:
            x = x + self.bias
        # if self.bn is not None:
        #     x = self.bn(x)
        #     x = self.drop(x)
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x