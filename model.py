# DBN Architecture
import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, visible, hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(visible, hidden) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(visible))
        self.h_bias = nn.Parameter(torch.zeros(hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(v @ self.W + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return prob, torch.bernoulli(prob)

class DBN(nn.Module):
    def __init__(self, layers=[64, 32, 16]):
        super().__init__()
        self.rbms = nn.ModuleList([RBM(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.classifier = nn.Linear(layers[-1], 1)

    def forward(self, x):
        for rbm in self.rbms:
            prob, _ = rbm.sample_h(x)
            x = prob
        return self.classifier(x)
