import torch.nn as nn
from src.supports.kan import KAN
from torch.nn.modules.activation import MultiheadAttention
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2
# from mamba_ssm import Mamba
import torch
from src.supports.constants import Constants
from src.supports.rms_norm import RMSNorm
from src.supports.flatten_head import Flatten_Head


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.lr = Constants(configs.dataset).get_learning_rate()
        self.encoders = nn.ModuleList([Encoder(configs) for _ in range(configs.e_layers)])
        self.reconstructor = Reconstructor(configs)

    def forward(self, x):
        for encoder in self.encoders:
            x1, x = encoder(x)
        x = self.reconstructor(x)
        return x, x


class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.d_model = configs.d_model
        self.features = configs.features
        self.window_size = configs.window_size
        self.batch_size = configs.batch_size

        self.linear1 = nn.Linear(self.features, self.d_model)
        self.linear2 = nn.Linear(self.d_model, self.features)
        self.attn = MultiheadAttention(self.d_model, num_heads=configs.n_heads)
        self.dropout = nn.Dropout(configs.dropout)
        self.layernorm = nn.ModuleList([RMSNorm(self.d_model, eps=1e-5) for _ in range(3)])
        self.kan1 = KAN([self.window_size * self.features, self.batch_size, self.d_model * self.window_size], seed=configs.seed)
        # self.mamba1 = Mamba(d_model=self.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)

    def forward(self, x):
        kan_output = self.kan1(x.reshape(x.shape[0], -1)).reshape(-1, self.window_size, self.d_model)
        x = self.dropout(kan_output) + self.linear1(x)
        x = self.layernorm[1](x)
        x1 = x

        attn_output = self.attn(x, x, x)[0]
        x = self.dropout(attn_output) + x
        x = self.layernorm[0](x)

        # x = self.mamba1(x)
        # x = self.dropout(x) + x
        # x = self.layernorm[2](x)

        x = self.linear2(x)
        return x1, x


class Reconstructor(nn.Module):
    def __init__(self, configs):
        super(Reconstructor, self).__init__()
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch

        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.flatten = Flatten_Head(configs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.W_P(x)
        x = x.permute(0, 1, 3, 2)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x
