import torch
from torch import nn
from importlib import import_module

class Model(nn.Module):
    def __init__(self, config, env):
        super().__init__()

    def forward(self):
        pass

    def get_action(self, state) -> torch.Tensor:
        pass

    def get_loss(self, *frame) -> torch.Tensor:
        pass

def get_model(config, env) -> Model:
    model_module = import_module(f"..{config.name}", package=__name__)
    model = getattr(model_module, config.name)(config, env)

    return model