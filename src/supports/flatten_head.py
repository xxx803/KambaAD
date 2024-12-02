import torch
from torch import nn
import torch.nn.init as init


class Flatten_Head(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.individual = configs.individual
        self.n_vars = configs.enc_in
        target_window = configs.window_size
        head_dropout = configs.head_dropout
        d_model = configs.d_model
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        context_window = configs.window_size

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        # print(f'patch_num: {patch_num},context_window: {context_window},stride: {stride}')
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        nf = d_model * patch_num
        # Backbone
        # Head
        self.n_vars = configs.enc_in
        self.head_nf = d_model * patch_num

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()
        for i in range(self.n_vars):
            self.flattens.append(nn.Flatten(start_dim=-2))

            linear_layer = nn.Linear(nf, target_window)

            init.kaiming_normal_(linear_layer.weight)
            # 这里可以换成xavier_normal_，kaiming_uniform_，kaiming_normal_，
            # 或者with torch.no_grad():
            #     linear_layer.weight.fill_(1/patchnum)
            init.zeros_(linear_layer.bias)
            self.linears.append(linear_layer)
            self.dropouts.append(nn.Dropout(head_dropout))

    def forward(self, x):
        x_out = []
        for i in range(self.n_vars):
            z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
            z = self.linears[i](z)  # z: [bs x target_window]
            z = self.dropouts[i](z)
            x_out.append(z)
        x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        return x
