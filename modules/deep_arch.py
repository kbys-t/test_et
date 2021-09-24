# coding:utf-8

import sys
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

###############################################################################
def get_ActivationFunc(name):
    # parameters are default
    lists = {
            # relu relatives
            "relu": F.relu,
            "celu": F.celu,
            "softplus": F.softplus,
            # sigmoid relatives
            "sigmoid": torch.sigmoid,
            "softsign": F.softsign,
            "tanh": torch.tanh,
            # shrink relatives
            "hardshrink": F.hardshrink,
            "softshrink": F.softshrink,
            "tanhshrink": F.tanhshrink,
            # handmade
            "swish": lambda x: x * torch.sigmoid(x),
            "mish": lambda x: x * (torch.tanh(F.softplus(x))),
            }
    if name in lists:
        return lists[name]
    else:
        print("{} is not implemented: {}".format(name, lists))
        sys.exit()

###############################################################################
def check_OutputSize(model, x_dim):
    with torch.no_grad():
        x = torch.zeros(x_dim).unsqueeze_(0)
        x = model(x)
        o_dim = tuple(x.size())[1:]
    return o_dim

###############################################################################
class Xception(nn.Module):
    def __init__(self, i_dim, out_channels, kernel_size, stride, transpose):
        super(Xception, self).__init__()
        #
        in_channels = i_dim[0]
        pointfirst = not(transpose)
        # modules
        cnn = nn.ConvTranspose2d if transpose else nn.Conv2d
        if pointfirst:
            # pointwise -> depthwise
            self.cnns = nn.Sequential(cnn(in_channels, out_channels, kernel_size=1, stride=1),
                                    cnn(out_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=out_channels)
                                    )
        else:
            # depthwise -> pointwise
            self.cnns = nn.Sequential(cnn(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels),
                                    cnn(in_channels, out_channels, kernel_size=1, stride=1)
                                    )
        self.o_dim = check_OutputSize(self, i_dim)

    def forward(self, x):
        return self.cnns(x)

######################################################
class ConvNet(nn.Module):
    def __init__(self, i_dim, channels=[8, 16, 32, 64, 128], kernel=5, stride=2, lnorm=False, a_name="relu"):
        super(ConvNet, self).__init__()
        chs = [channels] if isinstance(channels, int) else channels
        self.stride = stride
        self.lnorm = lnorm
        self.kernels = []
        layers = []
        norms = []
        for c in chs:
            idim = layers[-1].o_dim if len(layers) > 0 else i_dim
            self.kernels.append(kernel - ((idim[1] - kernel) % stride))   # assume: H and W are the same size
            layers.append(Xception(idim, c, self.kernels[-1], stride, False))
            norms.append(nn.LayerNorm(layers[-1].o_dim, elementwise_affine=lnorm) if lnorm is not None else nn.Identity())
        self.deep_net = nn.ModuleList(layers)
        self.deep_norm = nn.ModuleList(norms)
        self.o_dim = np.prod(layers[-1].o_dim)
        #
        self.fnc_activation = get_ActivationFunc(a_name)

    def forward(self, x):
        h = x
        for f, n in zip(self.deep_net, self.deep_norm):
            h = self.fnc_activation(n(f(h)))
        return h.view(len(h), -1)

######################################################
class FCNet(nn.Module):
    def __init__(self, i_dim, h_dims=[100]*3, lnorm=False, a_name="relu"):
        super(FCNet, self).__init__()
        # network structure
        ih_dims = [i_dim] + [h_dims] if isinstance(h_dims, int) else [i_dim] + h_dims
        self.o_dim = ih_dims[-1]
        layers = []
        norms = []
        for i in range(len(ih_dims)-1):
            layers.append(nn.Linear(ih_dims[i], ih_dims[i+1]))
            norms.append(nn.LayerNorm(ih_dims[i+1], elementwise_affine=lnorm) if lnorm is not None else nn.Identity())
        self.deep_net = nn.ModuleList(layers)
        self.deep_norm = nn.ModuleList(norms)
        #
        self.fnc_activation = get_ActivationFunc(a_name)

    def forward(self, x):
        h = x
        for f, n in zip(self.deep_net, self.deep_norm):
            h = self.fnc_activation(n(f(h)))
        return h
