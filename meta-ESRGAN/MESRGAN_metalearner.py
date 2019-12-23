#empirical loss
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from RRDBNet_arch import *
#from MESRGAN_generator import *

'''
StellEdge:
meta-SGD model

output (theta,alpha)  get loss-Ttest(theta)


θ represents the state of a learner that can be used to initialize the learner for
any new task, and α is a vector of the same size as θ 

while not None do:
    for each batch T contains (x,y) pairs:
        Ltrain(Ti)(θ) ← mean(loss(fθ(x), y))
        θi' ← θ − α ◦∇Ltrain(Ti)(θ)
        Ltest(Ti)(θi') ← mean(loss(fθi'(x), y))
    (θ, α) ← (θ, α) − β∇(θ,α)sigma-Ti(Ltest(Ti)(θi'))

θ and α are (meta-)parameters of the meta-learner to be learned, and ◦ denotes element-wise
product.

'''

N_FILTERS = 64  # number of filters used in conv_block
K_SIZE = 3  # size of kernel
MP_SIZE = 2  # size of max pooling
EPS = 1e-8  # epsilon for numerical stability


class MetaLearner(nn.Module):
    """
    The class defines meta-learner for Meta-SGD algorithm.
    """
    def __init__(self, params):
        super(MetaLearner, self).__init__()
        self.params = params
        self.meta_learner = Net(
            params.channels,3, 64, opt.n_residual_blocks, gc=32,bias=False, dataset=params.dataset_name)
        #G=RRDBNet(3, 3, 64, opt.n_residual_blocks, gc=32) #origin 23 blocks
        self.task_lr = OrderedDict()

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X) 
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def cloned_state_dict(self):
        """
        Only returns state_dict of meta_learner (not task_lr)
        """
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

    def define_task_lr_params(self):
        for key, val in self.named_parameters():
            # self.task_lr[key] = 1e-3 * torch.ones_like(val, requires_grad=True)
            self.task_lr[key] = nn.Parameter(
                1e-3 * torch.ones_like(val, requires_grad=True))
