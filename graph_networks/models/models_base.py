import torch
import torch.nn as nn 

class Parallel(nn.Sequential):
    def __init__(self, *args) -> None:
        super().__init__(*args) 

    def forward(self, x, channel_dim=-1):
        out= []
        for idx, module in enumerate(self):
            # dump= torch.index_select(x, channel_dim, torch.tensor(idx))
            out.append(module(x[..., idx]))
        return torch.stack(out, dim= channel_dim)