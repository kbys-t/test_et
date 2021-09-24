# coding:utf-8

import torch
from torch.distributions.kl import register_kl

from torch.distributions.studentT import StudentT

from modules.distributions.gumbel_categorical import GumbelCategorical
from modules.distributions.gumbel_onehot_categorical import GumbelOneHotCategorical
from modules.distributions.multivariate_diag_studentT import MultivariateDiagStudentT

### categorical distributions
# copied from pytorch
@register_kl(GumbelCategorical, GumbelCategorical)
def _kl_gumbelcategorical_gumbelcategorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = float("inf")
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)

@register_kl(GumbelOneHotCategorical, GumbelOneHotCategorical)
def _kl_gumbelonehotcategorical_gumbelonehotcategorical(p, q):
    return _kl_gumbelcategorical_gumbelcategorical(p._categorical, q._categorical)

### student-t distributions
# approximation of KLD b/w student-t distributions
# as normal-gamma distributions
# https://arxiv.org/pdf/1611.01437.pdf
@register_kl(StudentT, StudentT)
def _kl_student_student(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    ddf = (q.df * (p.df.log() - q.df.log())
        - 2.0 * ((0.5 * p.df).lgamma() - (0.5 * q.df).lgamma())
        + (p.df - q.df) * ((0.5 * p.df).digamma() - 1.0))
    return 0.5 * (var_ratio + t1 - 1.0 - var_ratio.log() + ddf)

@register_kl(MultivariateDiagStudentT, MultivariateDiagStudentT)
def _kl_multivariatediagstudent_multivariatediagstudent(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    ddf = 0.5 * (q.df * (p.df.log() - q.df.log())
        - 2.0 * ((0.5 * p.df).lgamma() - (0.5 * q.df).lgamma())
        + (p.df - q.df) * ((0.5 * p.df).digamma() - 1.0))
    return 0.5 * (var_ratio + t1 - 1.0 - var_ratio.log()).sum(-1) + ddf
