from torch import nn
from PROSE_HGNN.models import HGNN_conv
import torch.nn.functional as F
import torch

class HGNN(nn.Module):
    # def __init__(self, in_ch, n_hid, n_class, dropout=0.5):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, dropout_adj, sparse, batch_norm):
        super(HGNN, self).__init__()
        # self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_channels, hidden_channels)
        self.hgc2 = HGNN_conv(hidden_channels, out_channels)
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.act = nn.ReLU(inplace=True)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
    def forward(self, x, G):
        # x = x.to(torch.float64)
        x = x.to(torch.float16)
        # G = self.dropout_adj(G)
        G = G.to(torch.float16)
        x = self.act(self.hgc1(x, G))
        if self.batch_norm:
            x = self.bn1(x)

        x = self.hgc2(x, G)

        return x
