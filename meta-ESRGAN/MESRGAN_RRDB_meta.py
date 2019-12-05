import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

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
BIAS=False

class ResidualDenseBlock_meta(nn.Module):
    def __init__(self, channel_in=64, channel_gain=32,in_block_layer=4,res_alpha=0.2, bias=False):
        super(ResidualDenseBlock_meta, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv_layer_list=[]
        self.in_block_layer=in_block_layer
        self.res_alpha=res_alpha 
        for i in range(in_block_layer):
            self.conv_layer_list.append(nn.Conv2d(channel_in+i*channel_gain,channel_gain, 3, 1, 1, bias=bias))
        self.conv_layer_list.append(nn.Conv2d(channel_in+in_block_layer*channel_gain,  channel_in, 3, 1, 1, bias=bias))
        self.conv_layer_list=nn.ModuleList(self.conv_layer_list)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x_res=[]
        x_res.append(x)
        for i in range(self.in_block_layer):
            x_res.append( self.lrelu(self.conv_layer_list[i](torch.cat(x_res, 1))))
        x_out = self.conv_layer_list[-1](torch.cat(x_res, 1))
        return x_out * self.res_alpha + x

    def forward_meta(self,x,params,trunk_index,block_index):
        x_res=[]
        x_res.append(x)
        state_dict_param_pre='RRDB_trunk.'+str(trunk_index)+'.blocks.'+str(block_index)+'.conv_layer_list.'
        for i in range(self.in_block_layer):
            state_dict_param=state_dict_param_pre+str(i)
            if(BIAS):
                temp_middle=F.conv2d(torch.cat(x_res,1),
                                    params[state_dict_param+'.weight'],
                                    params[state_dict_param+'.bias'],
                                    padding=1)
            else:
                temp_middle=F.conv2d(torch.cat(x_res,1),
                                    params[state_dict_param+'.weight'],
                                    padding=1)
            x_res.append(F.leaky_relu(temp_middle))
        if(BIAS):
            x_out = F.conv2d(torch.cat(x_res, 1),
                             params[state_dict_param_pre+'4.weight'],
                             params[state_dict_param_pre+'4.bias'],
                                    padding=1)
        else:
            x_out = F.conv2d(torch.cat(x_res, 1),
                             params[state_dict_param_pre+'4.weight'],
                                    padding=1)
        return x_out * self.res_alpha + x

class RRDB_meta(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self,  channel_in=64, channel_gain=32,block_num=3,res_alpha=0.2):
        super(RRDB_meta, self).__init__()
        self.blocks=[]
        self.block_num=block_num
        self.res_alpha=res_alpha
        for _ in range(block_num):
            self.blocks.append(ResidualDenseBlock_meta( channel_in, channel_gain))
        self.blocks=nn.ModuleList(self.blocks)
    def forward(self, x):
        out=x
        for i in range(self.block_num):
            out=self.blocks[i](out)
        return out * self.res_alpha + x

    def forward_meta(self,x,params,trunk_index):
        out=x
        for i in range(self.block_num):
            out=self.blocks[i].forward_meta(out,params,trunk_index,i)
        return out * self.res_alpha + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class RRDBNet_meta(nn.Module):
    def __init__(self, channel_in, channel_out, channel_flow, block_num, gc=32,bias=False):
        super(RRDBNet_meta, self).__init__()
        RRDB_block_f = functools.partial(RRDB_meta,channel_in=channel_flow, channel_gain=gc)

        self.conv_first = nn.Conv2d(channel_in, channel_flow, 3, 1, 1, bias=bias)
        self.RRDB_trunk = make_layer(RRDB_block_f, block_num)
        self.trunk_conv = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        #### upsampling
        '''
        self.upsample=[
            F.interpolate(fea, scale_factor=2, mode='nearest'),
            nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            F.interpolate(fea, scale_factor=2, mode='nearest'),
            nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_flow, channel_out, 3, 1, 1, bias=True)
        ]
        '''
        
        self.upconv1 = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        self.upconv2 = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        self.HRconv = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        self.conv_last = nn.Conv2d(channel_flow, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, x,params=None):
        if params==None:
            fea = self.conv_first(x)
            trunk = self.trunk_conv(self.RRDB_trunk(fea))
            fea = fea + trunk

            #out=self.upsample(fea)
        
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.HRconv(fea)))        
            return out
        else:
            if(BIAS):
                fea=F.conv2d(x,
                            params['conv_first.weight'],
                            params['conv_first.bias'],
                                    padding=1)
            else:
                fea=F.conv2d(x,
                            params['conv_first.weight'],
                                    padding=1)

            trunk_index=0
            trunk=fea
            for layer in self.RRDB_trunk:
                trunk=layer.forward_meta(trunk,params,trunk_index)
                trunk_index+=1

            if(BIAS):
                trunk=F.conv2d(trunk,
                               params['trunk_conv.weight'],
                               params['trunk_conv.bias'],
                               padding=1)
                fea=fea+trunk
                fea=F.leaky_relu(F.conv2d(F.interpolate(fea, scale_factor=2, mode='nearest'),
                                          params['upconv1.weight'],
                                          params['upconv1.bias'],
                                          padding=1))
                fea=F.leaky_relu(F.conv2d(F.interpolate(fea, scale_factor=2, mode='nearest'),
                                          params['upconv2.weight'],
                                          params['upconv2.bias'],
                                          padding=1))
                fea=F.leaky_relu(F.conv2d(fea,
                                          params['HRconv.weight'],
                                          params['HRconv.bias'],
                                          padding=1))
                out=F.conv2d(fea,
                             params['conv_last.weight'],
                             params['conv_last.bias'],
                             padding=1)
                return out
            else:
                trunk=F.conv2d(trunk,
                               params['trunk_conv.weight'],
                               padding=1)
                fea=fea+trunk
                fea=F.leaky_relu(F.conv2d(F.interpolate(fea, scale_factor=2, mode='nearest'),
                                          params['upconv1.weight'],
                                          padding=1))
                fea=F.leaky_relu(F.conv2d(F.interpolate(fea, scale_factor=2, mode='nearest'),
                                          params['upconv2.weight'],
                                          padding=1))
                fea=F.leaky_relu(F.conv2d(fea,
                                          params['HRconv.weight'],
                                          padding=1))
                out=F.conv2d(fea,
                             params['conv_last.weight'],
                             padding=1)
                return out




