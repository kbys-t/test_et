# coding:utf-8

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from modules.distributions.gumbel_onehot_categorical import GumbelOneHotCategorical
from modules.distributions.multivariate_diag_studentT import MultivariateDiagStudentT
from modules.distributions import kl

######################################################
def get_dist(name, i_dim, o_dim, **kwargs):
    rtv = None
    if "MD" in name:
        if "categorical" in name:
            rtv = MDCategoricalNet(i_dim, o_dim, **kwargs)
        elif "student" in name:
            rtv = MDStudentTNet(i_dim, o_dim, **kwargs)
        else:
            rtv = MDNormalNet(i_dim, o_dim, **kwargs)
    else:
        if "categorical" in name:
            rtv = CategoricalNet(i_dim, o_dim, **kwargs)
        elif "student" in name:
            rtv = StudentTNet(i_dim, o_dim, **kwargs)
        else:
            rtv = NormalNet(i_dim, o_dim, **kwargs)
    return rtv

######################################################
class MDNormalNet(nn.Module):
    def __init__(self, i_dim, o_dim, **kwargs):
        super(MDNormalNet, self).__init__()
        self.model = NormalNet(i_dim, o_dim, **kwargs)

    def forward(self, x):
        return torch.distributions.independent.Independent(self.model(x), 1)

######################################################
class MDCategoricalNet(nn.Module):
    def __init__(self, i_dim, o_dim, **kwargs):
        super(MDCategoricalNet, self).__init__()
        self.model = CategoricalNet(i_dim, o_dim, **kwargs)

    def forward(self, x):
        return torch.distributions.independent.Independent(self.model(x), 1)

######################################################
class NormalNet(nn.Module):
    def __init__(self, i_dim, o_dim, eps=0.0, **kwargs):
        super(NormalNet, self).__init__()
        self.loc = nn.Linear(i_dim, o_dim)
        self.scale = nn.Linear(i_dim, o_dim)
        #
        self.eps = eps

    def forward(self, x):
        return torch.distributions.normal.Normal(self.loc(x), F.softplus(self.scale(x)) + self.eps)

######################################################
class CategoricalNet(nn.Module):
    def __init__(self, i_dim, o_dim, temp=1.0, **kwargs):
        super(CategoricalNet, self).__init__()
        self.prob = nn.Linear(i_dim, o_dim)
        #
        self.temp = temp

    def forward(self, x):
        return GumbelOneHotCategorical(F.softmax(self.prob(x) / self.temp, dim=1))

######################################################
class StudentTNet(nn.Module):
    def __init__(self, i_dim, o_dim, eps=0.0, df0=2.0, **kwargs):
        super(StudentTNet, self).__init__()
        self.dof = nn.Linear(i_dim, 1)
        self.loc = nn.Linear(i_dim, o_dim)
        self.scale = nn.Linear(i_dim, o_dim)
        #
        self.eps = eps
        self.df0 = df0

    def forward(self, x):
        return torch.distributions.studentT.StudentT(F.softplus(self.dof(x)) + self.df0, self.loc(x), F.softplus(self.scale(x)) + self.eps)

######################################################
class MDStudentTNet(nn.Module):
    def __init__(self, i_dim, o_dim, eps=0.0, df0=2.0, **kwargs):
        super(MDStudentTNet, self).__init__()
        self.dof = nn.Linear(i_dim, 1)
        self.loc = nn.Linear(i_dim, o_dim)
        self.scale = nn.Linear(i_dim, o_dim)
        #
        self.eps = eps
        self.df0 = df0

    def forward(self, x):
        return MultivariateDiagStudentT(F.softplus(self.dof(x)) + self.df0, self.loc(x), F.softplus(self.scale(x)) + self.eps)
