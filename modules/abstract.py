# coding:utf-8

import numpy as np

import torch
from torch import nn, optim

from modules import optimizers as optim2

######################################################
class Abstract(nn.Module):
    def __init__(self,
                name="",
                ):
        super(Abstract, self).__init__()
        #
        self.name = name

###################
    def forward(self, x):
        return x

###################
    def init(self, params, opt_params, ldir, use_cuda):
        ops = opt_params.copy()
        method = ops.pop("method", "rmsprop")
        print("hoge", method)
        if method == "tadam":
            self.optimizer = optim2.TAdam(params, **ops)
        elif method == "tlaprop":
            self.optimizer = optim2.TLaProp(params, **ops)
        elif method == "adam":
            ops["k_dof"] = np.inf
            self.optimizer = optim2.TAdam(params, **ops)
        elif method == "laprop":
            ops["k_dof"] = np.inf
            self.optimizer = optim2.TLaProp(params, **ops)
        else:
            self.optimizer = optim.RMSprop(params, **ops)
        # load parameters
        print("parameters are initialized")
        print(self)
        print(self.optimizer)
        self.load(ldir)
        # send to gpu
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print("computed by {}".format(self.device))
        self.to(self.device)

###################
    def release(self, sdir):
        # save models
        torch.save(self.state_dict(), sdir + self.name + "_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + self.name + "_optim.pth")

###################
    def load(self, ldir):
        try:
            self.load_state_dict(torch.load(ldir + self.name + "_model.pth"))
            self.optimizer.load_state_dict(torch.load(ldir + self.name + "_optim.pth"))
            print("load parameters are in "+ldir)
            return True
        except:
            print("parameters are not loaded")
            return False

###################
    def convert(self, x, dim):
        rtv = None
        if "Tensor" in str(type(x)):
            # assume: one or three dimensional, and if three, x is already CxHxW
            rtv = x.view(-1, dim) if isinstance(dim, int) else x.view(-1, dim[0], dim[1], dim[2])
        else:
            # assume: one or three dimensional, and if three, x is HxWxC
            tmp = np.array(x).reshape((-1, dim)) if isinstance(dim, int) else np.array(x).reshape((-1, dim[1], dim[2], dim[0])).transpose(0, 3, 1, 2)
            rtv = torch.from_numpy(tmp.astype("float32"))
        return rtv.to(self.device)
