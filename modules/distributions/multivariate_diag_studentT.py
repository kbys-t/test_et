import math

import torch
from torch._six import inf, nan
from torch.distributions import Chi2, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal


class MultivariateDiagStudentT(Distribution):
    r"""
    Creates a Multivariate Diagonal Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.

    Example::

        >>> m = MultivariateDiagStudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
        tensor([ 0.1046])

    Args:
        df (Tensor[batch_shape, 1]): degrees of freedom
        loc (Tensor[batch_shape, dim]): mean of the distribution
        scale (Tensor[batch_shape, dim]): scale of the distribution
    """
    arg_constraints = {'df': constraints.positive, 'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        m = self.loc.clone()
        m[self.df <= 1, :] = nan
        return m

    @property
    def variance(self):
        v = self.scale.pow(2)
        v[self.df > 2, :] *= self.df[self.df > 2, :] / (self.df[self.df > 2, :] - 2)
        v[(self.df <= 2) & (self.df > 1), :] = inf
        v[self.df <= 1, :] = nan
        return v

    def __init__(self, df, loc, scale, validate_args=None):
        dim = loc.size(-1)
        if scale.size(-1) != dim:
            raise ValueError("scale should have the same dimension as loc")
        shape = loc.reshape(-1, dim).shape
        batch_shape, event_shape = shape[:-1], shape[-1:]
        self.df = df.expand(batch_shape) if df.dim() == 1 else df.squeeze(-1).expand(batch_shape)
        self._chi2 = Chi2(self.df)
        self.loc = loc.expand(batch_shape + event_shape)
        self.scale = scale.expand(batch_shape + event_shape)
        super(MultivariateDiagStudentT, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateDiagStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self._event_shape
        new.df = self.df.expand(batch_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        new.loc = self.loc.expand(loc_shape)
        new.scale = self.scale.expand(loc_shape)
        super(MultivariateDiagStudentT, new).__init__(batch_shape, self._event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df).unsqueeze(-1)
        return self.loc + self.scale * Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dim = self.loc.size(-1)
        y = (value - self.loc) / self.scale
        Z = (self.scale.log().sum(-1) +
             0.5 * dim * (self.df * math.pi).log() +
             torch.lgamma(0.5 * self.df) -
             torch.lgamma(0.5 * (self.df + dim)))
        return -0.5 * (self.df + dim) * torch.log1p(y.pow(2).sum(-1) / self.df) - Z

    def entropy(self):
        dim = self.loc.size(-1)
        return (self.scale.log().sum(-1) +
                0.5 * (self.df + dim) *
                (torch.digamma(0.5 * (self.df + dim)) - torch.digamma(0.5 * self.df)) +
                0.5 * dim * (self.df * math.pi).log() +
                torch.lgamma(0.5 * self.df) - torch.lgamma(0.5 * (self.df + dim)))
