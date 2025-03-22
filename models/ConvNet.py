__all__ = ['ConvNet.py']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
# from layers.PatchTST_layers import *
# from layers.RevIN import RevIN

class PatchMixerLayer(nn.Module):

    def __init__(self,patch_size,dim,a,kernel_size = 7,):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.Resnet =  nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, groups=dim,padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim ,a,kernel_size=1),
            nn.ReLU(),
        )

    def forward(self,x):
        a = self.Resnet(x)
        x = x + a                 # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len,patch_len,stride,mixer_kernel_size,d_model,dropout,head_dropout,e_layers):
        super().__init__()
        self.model = Backbone(enc_in, seq_len, pred_len,patch_len,stride,mixer_kernel_size,d_model,dropout,head_dropout,e_layers)
    def forward(self, x):
        x = x.float()
        x = self.model(x)
        return x
class Backbone(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len,patch_len,stride,mixer_kernel_size,d_model,dropout,head_dropout,e_layers,revin = True, affine = True, subtract_last = False):
        super().__init__()

        self.nvals = enc_in
        self.lookback = seq_len
        self.forecasting = pred_len
        self.patch_size = patch_len
        self.stride = stride
        self.kernel_size = mixer_kernel_size
        self.Conv = nn.Sequential(
            nn.Conv1d(self.patch_size,2,kernel_size=1),
        )
        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        if self.stride==0:
            self.patch_num = int(self.lookback/self.patch_size)
        else:
            self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num
        self.d_model = d_model
        self.dropout = 0.2
        self.head_dropout = head_dropout
        self.depth = e_layers
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(patch_size=self.patch_size,dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))

        self.head0 = nn.Sequential(
            nn.MaxPool1d(kernel_size=1, stride=2),
            nn.Flatten(start_dim=-2),

        )

        self.dropout = nn.Dropout(self.dropout)
        self.head_dropout = nn.Dropout(self.head_dropout)


    def forward(self, x):

        # print(x.shape)
        # print(x.device)
        x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]
        x = np.squeeze(x, axis=1)

        for PatchMixer_block in self.PatchMixer_blocks:

            x = PatchMixer_block(x)

        x = x.permute(0, 2, 1)
        x = self.Conv(x)
        x = self.head_dropout(x)

        x = self.head0(x)

        return x