import torch
import torch.nn as nn
import torch.nn.functional as F

'''
blocks for RRDB ESRGAN basic
'''
class ResidualDenseBlock(nn.Module):
    def __init__(self, channel_in=64, channel_gain=32,in_block_layer=4,res_alpha=0.2, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv_layer_list=[]
        self.in_block_layer=in_block_layer
        self.res_alpha=res_alpha 
        for i in range(in_block_layer):
            self.conv_layer_list.append(nn.Conv2d(channel_in+i*channel_gain,channel_gain, 3, 1, 1, bias=bias))
        self.conv_layer_list.append(nn.Conv2d(channel_in+in_block_layer*channel_gain,  channel_in, 3, 1, 1, bias=bias))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x_res=[]
        for i in in_block_layer:
            x_res.append( self.lrelu(self.conv_layer_list[i](torch.cat(x_res, 1))))
        x_out = self.conv_layer_list[-1](torch.cat(x_res, 1))
        return x_out * res_alpha + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self,  channel_in=64, channel_gain=32,block_num=3,res_alpha=0.2):
        super(RRDB, self).__init__()
        self.blocks=[]
        self.block_num=block_num
        self.res_alpha=res_alpha
        for _ in range(block_num):
            self.blocks.append(ResidualDenseBlock( channel_in, channel_gain))
    def forward(self, x):
        out=x
        for i in range(block_num):
            out=self.blocks[i](out)
        return out * res_alpha + x

