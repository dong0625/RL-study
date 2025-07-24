import torch
from torch import nn
from importlib import import_module

class Model(nn.Module):
    def __init__(self, args, env):
        super().__init__()

    def forward(self):
        pass

    def get_action(self, state) -> torch.Tensor:
        pass

    def get_loss(self, frame) -> torch.Tensor:
        pass

def get_model(model_name : str, args, env) -> Model:
    model_module = import_module(model_name)
    model = model_module.Model(args, env)

    return model