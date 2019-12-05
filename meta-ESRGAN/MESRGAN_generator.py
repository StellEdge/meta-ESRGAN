import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
'''
blocks for RRDB ESRGAN basic
'''
class ResidualDenseBlock(nn.Module):
    def __init__(self, channel_in=64, channel_gain=32,in_block_layer=4,res_alpha=0.2, bias=False):
        super(ResidualDenseBlock, self).__init__()
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

class RRDB(nn.Module):
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

class RRDBNet(nn.Module):
    def __init__(self, channel_in, channel_out, channel_flow, block_num, gc=32,bias=False):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB,channel_in=channel_flow, channel_gain=gc)

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
        

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        #out=self.upsample(fea)
        
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))        
        return out
    def get_RRDB_out_features(self,x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        out = self.conv_last(fea)        
        return out