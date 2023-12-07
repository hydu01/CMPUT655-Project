import numpy as np
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
    def __init__(self, sigma=2.0, d=20.0):
        super().__init__()
        self.sigma = sigma
        self.d = d
    
    def forward(self, x):
        return 1.0/(1.0+torch.pow(torch.abs(x/self.sigma), self.d))


class MLP(nn.Module):
    def __init__(self, dims, last_layer_bias=None):
        super().__init__()
        self.model = self.make_model(dims, last_layer_bias)
    
    def make_model(self, dims, last_layer_bias):
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if last_layer_bias is not None:
                with torch.no_grad():
                    layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class EMLP(nn.Module):
    def __init__(self, dims, last_layer_bias="elephant"):
        super().__init__()
        self.model = self.make_model(dims, last_layer_bias)
    
    def make_model(self, dims, last_layer_bias):
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            with torch.no_grad():
                layers[-1].bias.uniform_(-1.0, 1.0)
            if i == len(dims) - 2:
                layers.append(Elephant())
            else:
                layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        with torch.no_grad():
            if last_layer_bias == "elephant":
                layers[-1].bias.uniform_(-1.0, 1.0)
            elif last_layer_bias is not None:
                layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        

class CNN(nn.Module):
    def __init__(
        self,
        obs_width,
        channels,
        hidden_dims,
        out_dim, 
        last_layer_bias = None
    ):
        super(CNN, self).__init__()
        hidden_dims.append(out_dim)
        self.model = self.make_model(obs_width,
                                     channels,
                                     hidden_dims,
                                     last_layer_bias)

    def make_model(self, obs_width, channels, hidden_dims, last_layer_bias):
        layers = []
        k = 2       # Kernel size
        s = 1       # Stride
        p = 0       # Padding
        m = False   # Max pooling
        for i in range(len(channels) - 1):
            # Add Convolutional layers
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=k, stride=s))
            obs_width, k, s, p, m = flatten(obs_width, k, s, p, m)

        # Flatten the CNN output amd compute the flattened dim size
        layers.append(nn.Flatten())
        flatten_dim_size = obs_width * obs_width * channels[-1]
        hidden_dims.insert(0, flatten_dim_size)

        for i in range(len(hidden_dims)-2):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        if last_layer_bias is not None:
                with torch.no_grad():
                    layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def flatten(w, k=3, s=1, p=0, m=True):
    # Reference: https://stackoverflow.com/questions/59108988/flatten-tensor-in-pytorch-convolutional-neural-network-size-mismatch-error
    return int((np.floor((w - k + 2 * p) / s) + 1) / 2 if m else 1), k, s, p, m