import torch
import torch.nn as nn

from .models_base import Parallel

class SimpleAnn(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, out_channels) -> None:
        super().__init__()
        models = [nn.Sequential(nn.Linear(in_size, hidden_size),
                                nn.LeakyReLU(0.2),
                                nn.Linear(hidden_size, 10*hidden_size), 
                                nn.LeakyReLU(0.2),
                                nn.Linear(10*hidden_size, out_size)) for _ in range(out_channels)]
        self.model= Parallel(*models)

    def forward(self, x):
        return self.model(x)

