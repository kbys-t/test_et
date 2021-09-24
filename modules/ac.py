# coding:utf-8

import copy

import torch
from torch import nn, optim
from torch.nn import functional as F

from modules.abstract import Abstract
from modules import deep_arch
from modules import dist_arch

######################################################
class AC(Abstract):
    def __init__(self, s_dim, a_dim, is_discrete,
                cnn_params={"channels": [8, 16, 32, 64, 128], "kernel": 5, "stride": 2},
                fc_params={"h_dims": [128]*5},
                opt_params={"lr": 1e-4, "amsgrad": 0.99999, "method": "tlaprop"},
                gamma=0.99, lambda1=0.5, lambda2=0.9, div_weight=1e+1,
                tar_tau=1.0, tar_dof=float("inf"),
                ppo_threshold=0.1,
                reg_weight=1e-1, reg_balance=0.5,
                dist="student", a_name="swish", eps=1e-8,
                use_cuda=False, ldir=None
        ):
        super(AC, self).__init__("ac")
        # store config of state action space
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.is_discrete = is_discrete
        # hyperparameters
        self.epsilon = eps
        self.discount_factor = gamma
        self.eligibility_rate1 = lambda1
        self.eligibility_rate2 = lambda2
        self.div_weight = div_weight
        self.div_sum = 0.0
        # ppo-like regularization
        self.ppo_threshold = ppo_threshold
        # experimental: TD-Ent regularization
        self.reg_weight = reg_weight
        self.reg_balance = reg_balance
        # network structure
        nns = {}
        # cnn
        if not isinstance(s_dim, int):
            nns["cnn"] = deep_arch.ConvNet(s_dim, a_name=a_name, **cnn_params)
            s_dim = nns["cnn"].o_dim
        # value func
        tmpnetv = deep_arch.FCNet(s_dim, a_name=a_name, **fc_params)
        nns["value"] = nn.Sequential(tmpnetv, nn.Linear(tmpnetv.o_dim, 1))
        # policy
        tmpnetp = deep_arch.FCNet(s_dim, a_name=a_name, **fc_params)
        nns["policy"] = nn.Sequential(tmpnetp, dist_arch.get_dist("MD-" + ("categorical" if is_discrete else dist), tmpnetp.o_dim, a_dim, eps=self.epsilon))
        self.nets = nn.ModuleDict(nns)
        # common initialization
        self.init(self.nets.parameters(), opt_params, ldir, use_cuda)
        # eligibility traces
        self.eligibility_trace = [torch.zeros_like(p_.data) for p_ in self.nets.parameters()]
        self.eligibility_replace = [torch.zeros_like(p_.data) for p_ in self.nets.parameters()]

###################
    def forward(self, x):
        h = self.convert(x, self.s_dim)
        if "cnn" in self.nets:
            h = self.nets["cnn"](h)
        pi = self.nets["policy"](h)
        value = self.nets["value"](h)
        return (pi.rsample() if self.training else pi.mean), pi, value

###################
    def criterion(self, policy, base_policy, action, value, base_value, next_value, reward, done):
        # variables to be used
        log_pi = policy.log_prob(action.data)
        balance_reward_reg_choice = self.reg_weight * self.reg_balance
        balance_tdvar_qent_choice = self.reg_balance
        with torch.no_grad():
            # adaptive decaying factor for eligibility traces
            decay = torch.ones(1, dtype=torch.float32, device=self.device)
            if self.div_weight:
                # set divergence as Bregman divergences
                # for policy as kld (kld is positive in theory, but approximated ones would not satisfy that)
                div_policy = F.relu(torch.distributions.kl.kl_divergence(policy, base_policy).mean())
                # for value as mse
                div_value = 0.5 * (value - base_value).pow(2).mean()# * self.a_dim
                # total divergence
                self.div_sum += (div_policy + div_value)
                # set decaying factor
                decay = (- self.div_weight * self.div_sum).exp()
                # decay sum
                self.div_sum *= decay
            # compute TD error (with normalirzed reward)
            r_ = torch.zeros_like(value) + reward
            # norm_factor = 1.0
            norm_factor = (1.0 - self.discount_factor)
            tderr = (norm_factor * (1.0 - balance_reward_reg_choice) * r_ + (1.0 - done) * self.discount_factor * next_value - value).sum(dim=1)
            r_ = r_.sum(dim=1)
            # TD-Ent regularization
            if self.reg_weight:
                tderr_raw = tderr / norm_factor
                reg_td = tderr_raw.pow(2)
                reg_pi = log_pi
                # surrogate TD error
                reg_all = - (balance_tdvar_qent_choice * reg_td + (1.0 - balance_tdvar_qent_choice) * reg_pi)
                tderr += norm_factor * balance_reward_reg_choice * reg_all
            # relative importance sampling (yamada+ 2011)
            dp = log_pi.exp()
            db = base_policy.log_prob(action.data).exp()
            ratio = dp / db
            # ppo-like regularization
            tdsign = tderr.sign()
            ratio = tdsign * torch.min(ratio * tdsign, ratio.clamp(1.0 - self.ppo_threshold, 1.0 + self.ppo_threshold) * tdsign)
        # compute main loss function for actor and critic
        loss_rl = (- tderr * ratio * (value + log_pi)).mean()
        #
        return tderr.mean(), decay, loss_rl

###################
    def update(self, tderr, decay, loss_rl):
        # reset gradients
        self.optimizer.zero_grad()
        # compute gradients for loss_rl w.r.t. nets
        loss_rl.backward()
        # set eligibility traces and reset gradients of nets
        for p_, e_, r_ in zip(self.nets.parameters(), self.eligibility_trace, self.eligibility_replace):
            if p_.grad is None:
                continue
            g_ = p_.grad / tderr
            g_[~torch.isfinite(g_)] = 0.0
            e_.mul_(self.discount_factor * self.eligibility_rate1 * decay).add_(g_)
            r_.mul_(self.discount_factor * self.eligibility_rate2 * decay)
            mask = (e_ - r_) * e_.sign() >= 0.0 if self.eligibility_rate1 else e_.abs() >= r_.abs()
            r_[mask] = e_[mask]
            p_.grad.zero_()
        # set gradients
        for p_, r_ in zip(self.nets.parameters(), self.eligibility_replace):
            if p_.grad is None:
                continue
            p_.grad += tderr * r_
        # update nets
        self.optimizer.step()

###################
    def act_sim2env(self, action, a_space=None):
        rtv = None
        if self.is_discrete:
            rtv = action.argmax().item()
        else:
            try:
                rtv = torch.sigmoid(action).cpu().data.numpy().flatten() * (a_space.high - a_space.low) + a_space.low
            except:
                rtv = action.cpu().data.numpy().flatten()
        return rtv

###################
    def reset(self):
        self.div_sum = 0.0
        for e_, r_ in zip(self.eligibility_trace, self.eligibility_replace):
            e_.zero_()
            r_.zero_()
