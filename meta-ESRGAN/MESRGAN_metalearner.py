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

class ResidualDenseBlock_indexed(nn.Module):
    def __init__(self,index, channel_in=64, channel_gain=32,in_block_layer=4,res_alpha=0.2, bias=False):
        super(ResidualDenseBlock_indexed, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv_layer_list=OrderedDict()
        self.in_block_layer=in_block_layer
        self.res_alpha=res_alpha 
        for i in range(in_block_layer):
            self.conv_layer_list.append(('name',nn.Conv2d(channel_in+i*channel_gain,channel_gain, 3, 1, 1, bias=bias)))
        self.conv_layer_list.append(nn.Conv2d(channel_in+in_block_layer*channel_gain,  channel_in, 3, 1, 1, bias=bias))
        self.conv_layer_list=nn.ModuleList(self.conv_layer_list)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_res=[]
        x_res.append(x)
        for i in range(self.in_block_layer):
            x_res.append( self.lrelu(self.conv_layer_list[i](torch.cat(x_res, 1))))
        x_out = self.conv_layer_list[-1](torch.cat(x_res, 1))
        return x_out * self.res_alpha + x

class RRDB_indexed(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self,  channel_in=64, channel_gain=32,block_num=3,res_alpha=0.2):
        super(RRDB, self).__init__()
        self.blocks=[]
        self.block_num=block_num
        self.res_alpha=res_alpha
        for _ in range(block_num):
            self.blocks.append(ResidualDenseBlock( channel_in, channel_gain))
        self.blocks=nn.ModuleList(self.blocks)
    def forward(self, x):
        out=x
        for i in range(self.block_num):
            out=self.blocks[i](out)
        return out * self.res_alpha + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

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

class Net(nn.Module):
    """
    RRDBNet for Meta-SGD for few-shot learning.
    """

    def __init__(self, channel_in, channel_out, channel_flow, block_num, gc=32,bias=False, dataset='div2k'):

        super(Net, self).__init__()
        '''
        self.features = nn.Sequential(
            conv_first(channel_in,channel_flow),
            RRDB_trunk(channel_flow,gc),
            else_blocks(channel_out,channel_flow)
            )
        '''
        self.conv_first = conv_first(channel_in,channel_flow)
        self.RRDB_trunk = make_layer(RRDB_block_f, block_num)
        self.trunk_conv = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
    def forward(self, X, params=None):

        if params == None:
            #normal forward


        else:
            """
            The architecure of functionals is the same as `self`.
            Here use F because no params will be saved.
            Only gradent will pass through. 
            """
            out=F.conv2d(
                X,
                params['meta_learner.features.0.conv_first.weight'],
                params['meta_learner.features.0.conv_first.bias'],
                padding=1)

            for trunk_index in range(23):
                for RDB_index in range(3):
                    pre="meta_learner.features.1."+str(trunk_index)+"."+str(RDB_index+1)+".RRDB_trunk_"+str(trunk_index)+"_RDB"+str(RDB_index+1)+"_conv"
                    for i in range(1,6):
                        out=F.conv2d(
                            out,
                            params[pre+str(i)+".weight"],
                            params[pre+str(i)+".bias"],
                            padding=1)
                    out=F.relu(out,inplace=True)

            out=F.conv2d(
                out,
                params['meta_learner.features.2.trunk_conv.weight'],
                params['meta_learner.features.2.trunk_conv.bias'],
                padding=1)

            out=F.conv2d(
                out,
                params['meta_learner.features.2.upconv1.weight'],
                params['meta_learner.features.2.upconv1.bias'],
                padding=1)

            out=F.conv2d(
                out,
                params['meta_learner.features.2.upconv2.weight'],
                params['meta_learner.features.2.upconv2.bias'],
                padding=1)

            out=F.conv2d(
                out,
                params['meta_learner.features.2.HRconv.weight'],
                params['meta_learner.features.2.HRconv.bias'],
                padding=1)

            out=F.conv2d(
                out,
                params['meta_learner.features.2.conv_last.weight'],
                params['meta_learner.features.2.conv_last.bias'],
                padding=1)

            out=F.relu(out,inplace=True)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['meta_learner.fc.weight'],
                           params['meta_learner.fc.bias'])

        out = F.log_softmax(out, dim=1)
        return out

def conv_first(in_nc,nf):
    block = nn.Sequential(OrderedDict([
            ('conv_first',nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True))
            ]))
    return block

def else_blocks(out_nc,nf):
    block = nn.Sequential(OrderedDict([
            ('trunk_conv',nn.Conv2d(nf,nf,3,1,1,bias=True)),
            ('upconv1',nn.Conv2d(nf,nf,3,1,1,bias=True)),
            ('upconv2',nn.Conv2d(nf,nf,3,1,1,bias=True)),
            ('HRconv',nn.Conv2d(nf,nf,3,1,1,bias=True)),
            ('conv_last',nn.Conv2d(nf,out_nc,3,1,1,bias=True)),
            ('lrelu', nn.LeakyReLU(negative_slope=0.2, inplace=True))
            ]))
    return block

def RRDB_trunk(nf,gc):
    """
    The RRDB in RRDBNet_arch.py
    The RRDB trunk contains 23 RRDB blocks.
    """
    trunk=nn.Sequential(
        RRDB_block(0,nf,gc),
        RRDB_block(1,nf,gc),
        RRDB_block(2,nf,gc),
        RRDB_block(3,nf,gc),
        RRDB_block(4,nf,gc),
        RRDB_block(5,nf,gc),
        RRDB_block(6,nf,gc),
        RRDB_block(7,nf,gc),
        RRDB_block(8,nf,gc),
        RRDB_block(9,nf,gc),
        RRDB_block(10,nf,gc),
        RRDB_block(11,nf,gc),
        RRDB_block(12,nf,gc),
        RRDB_block(13,nf,gc),
        RRDB_block(14,nf,gc),
        RRDB_block(15,nf,gc),
        RRDB_block(16,nf,gc),
        RRDB_block(17,nf,gc),
        RRDB_block(18,nf,gc),
        RRDB_block(19,nf,gc),
        RRDB_block(20,nf,gc),
        RRDB_block(21,nf,gc),
        RRDB_block(22,nf,gc)
        )
    return trunk

def RRDB_block(trunk_index,
               nf,
               gc=32):
    """
    A RRDB block(RRDB_trunk) contains 3 RDB block.
    Each RDB block contains 5 convolutional layer and 1 ReLU.
    """
    block=nn.Sequential(
        RDB_block(trunk_index,1,nf,gc),
        RDB_block(trunk_index,2,nf,gc),
        RDB_block(trunk_index,3,nf,gc)
        )
    return block

def RDB_block(trunk_index,
              RDB_index,
              nf=64,
              gc=32,
              bias=True):
    """
    A RDB block contains 5 convolutional layer and 1 ReLU.
    """
    pre="RRDB_trunk_"+str(trunk_index)+"_RDB"+str(RDB_index)+"_"
    block=nn.Sequential(
        OrderedDict([
            (pre+'conv1',nn.Conv2d(nf,gc,3,1,1,bias=bias)),
             (pre+'conv2',nn.Conv2d(nf+gc,gc,3,1,1,bias=bias)),
             (pre+'conv3',nn.Conv2d(nf+2*gc,gc,3,1,1,bias=bias)),
             (pre+'conv4',nn.Conv2d(nf+3*gc,gc,3,1,1,bias=bias)),
             (pre+'conv5',nn.Conv2d(nf+4*gc,gc,3,1,1,bias=bias)),
             (pre+'lrelu',nn.LeakyReLU(negative_slope=0.2, inplace=True))
             ]))
    return block

def conv_block(index,
               in_channels,
               out_channels=N_FILTERS,
               padding=0,
               pooling=True):
    """
    The unit architecture (Convolutional Block; CB) used in the modules.
    The CB consists of following modules in the order:
        3x3 conv, 64 filters
        batch normalization
        ReLU
        MaxPool
    """
    if pooling:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True)),
                ('pool'+str(index), nn.MaxPool2d(MP_SIZE))
            ]))
    else:
        conv = nn.Sequential(
            OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, \
                    K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, \
                    affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True))
            ]))
    return conv