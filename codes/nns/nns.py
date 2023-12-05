import torch
from torch import (
    nn,
    functional as F,
)


def make_model(model_config: dict):
    return


class Model(nn.Module):
    def __init__(self):
        pass


class Elephant(nn.Module):
    def __init__(self, sigma=1.0, d=2.0):
        super().__init__()
        self.sigma = sigma
        self.d = d
    
    def forward(self, x):
        return 1.0/(1.0+torch.power(torch.abs(x/self.sigma), self.d))


class MLP(nn.Module):
    def __init__(self, dims, last_layer_bias=10.0):
        super().__init__()
        self.model = make_model(dims, last_layer_bias)
    
    def make_model(self, dims, last_layer_bias):
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        with torch.no_grad():
            layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(layers)


class EMLP(nn.Module):
    def __init__(self, dims, last_layer_bias=10.0):
        super().__init__()
        self.model = make_model(dims, last_layer_bias)
    
    def make_model(self, dims, last_layer_bias):
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i == len(dims) - 2:
                layers.append(Elephant())
            else:
                layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        with torch.no_grad():
            layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(layers)
    
    def forward(self, x):
        return self.model(x)
        

            
