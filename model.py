import torch
from torch import nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim,
                    activation):
        super(GraphConvolution, self).__init__()

        self.activation = activation

        self.weight = nn.Parameter(nn.init.normal_(torch.Tensor(input_dim, output_dim), 0, 1 / 2 / output_dim))

    def forward(self, inputs):
        xs, supports = inputs
        xws = torch.matmul(xs, self.weight)
        outs = torch.matmul(supports, xws)
        if self.activation is None:
            return outs, supports
        else:
            return self.activation(outs), supports

class GCN(nn.Module):
    def __init__(self, deg_dim, input_dim, hidden_dim, layer_num = 2):
        super(GCN, self).__init__()

        self.deg_dim = deg_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential()
        self.layers.add_module('conv_first', GraphConvolution(self.input_dim, self.hidden_dim, F.relu))
        for i in range(layer_num - 2):
            self.layers.add_module('conv', GraphConvolution(self.hidden_dim, self.hidden_dim, F.relu))
        self.layers.add_module('conv_last', GraphConvolution(self.hidden_dim, self.hidden_dim, None))

    def forward(self, inputs):
        return torch.matmul(self.layers(inputs)[0], torch.ones(self.hidden_dim))

class CrossEntropy(nn.Module):
    def forward(self, output, labels):
        return torch.log(1 + torch.exp(-output*labels)).mean()