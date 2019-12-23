import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import skimage.color

'''
blocks for RRDB ESRGAN basic
'''
class ChannelAttention(nn.Module):
    '''
    each channel's feature means different in rebuilding HR image
    so by learning what features are being extracted,which feature matters most can be applied.
    '''
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        #for each channel,generate a meaningful alpha,
        #self attention mechanism with reduction synthesis
        self.input_size=128
        self.fea_extract = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 3,2, padding=2, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True), #64
                nn.Conv2d( channel//reduction,  channel//reduction, 3,2, padding=2, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True), #32
                nn.Conv2d( channel//reduction, channel, 3,2, padding=2, bias=True),
                #nn.LeakyReLU(negative_slope=0.2, inplace=True), #16
                nn.AdaptiveAvgPool2d(1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fea_extract(x)
        return x * y

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
        self.attention_layer=ChannelAttention(channel_in)
    def forward(self, x):
        out=x
        for i in range(self.block_num):
            out=self.blocks[i](out)
        res=out * self.res_alpha + x
        res=self.attention_layer(res)
        return res

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class RRDBNet(nn.Module):
    def __init__(self, channel_in, channel_out, channel_flow, block_num, gc=32,bias=True):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB,channel_in=channel_flow, channel_gain=gc)

        self.conv_first = nn.Conv2d(channel_in, channel_flow, 3, 1, 1, bias=bias)
        self.RRDB_trunk = make_layer(RRDB_block_f, block_num)
        self.trunk_conv = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        #### upsampling
     
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
    '''
    def get_RRDB_out_features(self,x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        out = self.conv_last(fea)        
        return out
    '''

class RRDBNet_shuffle(nn.Module):
    def __init__(self, channel_in, channel_out, channel_flow, block_num, gc=32,bias=True):
        super(RRDBNet_shuffle, self).__init__()
        RRDB_block_f = functools.partial(RRDB,channel_in=channel_flow, channel_gain=gc)

        self.conv_first = nn.Conv2d(channel_in, channel_flow, 3, 1, 1, bias=bias)
        self.RRDB_trunk = make_layer(RRDB_block_f, block_num)
        self.trunk_conv = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        #### upsampling
     
        self.channelconv1 = nn.Conv2d(channel_flow, channel_flow*4, 3, 1, 1, bias=bias)
        self.pixle_suffle1=nn.PixelShuffle(2)
        self.channelconv2 = nn.Conv2d(channel_flow, channel_flow*4, 3, 1, 1, bias=bias)
        self.pixle_suffle2=nn.PixelShuffle(2)
        
        #self.conv_deboard = nn.Conv2d(channel_flow, channel_flow, 5, 1, 2, bias=bias)
        self.conv_last = nn.Conv2d(channel_flow, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        #out=self.upsample(fea)
        
        fea = self.pixle_suffle1(self.lrelu(self.channelconv1(fea)))
        fea = self.pixle_suffle2(self.lrelu(self.channelconv2(fea)))
        out = self.conv_last(fea)
        return out

class RRDBNet_shuffle_flatten(nn.Module):
    def __init__(self, channel_in, channel_out, channel_flow, block_num, gc=32,bias=True):
        super(RRDBNet_shuffle_flatten, self).__init__()
        RRDB_block_f = functools.partial(RRDB,channel_in=channel_flow, channel_gain=gc)

        self.conv_first = nn.Conv2d(channel_in, channel_flow, 3, 1, 1, bias=bias)
        self.RRDB_trunk = make_layer(RRDB_block_f, block_num)
        self.trunk_conv = nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias)
        #### upsampling
     
        self.channelconv1 = nn.Conv2d(channel_flow, channel_flow*4, 3, 1, 1, bias=bias)
        self.pixle_suffle1=nn.PixelShuffle(2)
        self.channelconv2 = nn.Conv2d(channel_flow, channel_flow*4, 3, 1, 1, bias=bias)
        self.pixle_suffle2=nn.PixelShuffle(2)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(channel_flow, channel_flow, 3, 1, 1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_flow, channel_out, 1, 1, 0, bias=bias),
        )
        #self.conv_last = nn.Conv2d(channel_flow, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        #out=self.upsample(fea)
        
        fea = self.pixle_suffle1(self.lrelu(self.channelconv1(fea)))
        fea = self.pixle_suffle2(self.lrelu(self.channelconv2(fea)))
        out = self.conv_out(fea)
        return out

class HSVLoss(nn.Module):
    def __init__(self,H_coeff,S_coeff):
        super(HSVLoss,self).__init__()
        self.H_coeff=H_coeff
        self.S_coeff=S_coeff

    def convert_rgb_hsv(self,image):
        """Convert an RGB image to HSV.

        from kornia package
        https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv


        Args:
            input (torch.Tensor): RGB Image to be converted to HSV.

        Returns:
            torch.Tensor: HSV version of the image.
        """
        Tensor=torch.cuda.FloatTensor
        if not torch.is_tensor(image):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                             .format(image.shape))

        r: Tensor = image[..., 0, :, :]
        g: Tensor = image[..., 1, :, :]
        b: Tensor = image[..., 2, :, :]

        maxc: Tensor = image.max(-3)[0]
        minc: Tensor = image.min(-3)[0]

        v: Tensor = maxc  # brightness

        deltac: Tensor = maxc - minc
        s: Tensor = self.S_coeff*deltac / v  # saturation

        # avoid division by zero
        deltac: Tensor = torch.where(
            deltac == 0, torch.ones_like(deltac), deltac)

        rc: Tensor = (maxc - r) / deltac
        gc: Tensor = (maxc - g) / deltac
        bc: Tensor = (maxc - b) / deltac

        maxg: Tensor = g == maxc
        maxr: Tensor = r == maxc

        h: Tensor = 4.0 + gc - rc
        h[maxg]: Tensor = 2.0 + rc[maxg] - bc[maxg]
        h[maxr]: Tensor = bc[maxr] - gc[maxr]
        h[minc == maxc]: Tensor = 0.0

        h:Tensor = (self.H_coeff*h / 6.0) % 1.0
        out=torch.stack([h, s, v], dim=-3)
        out[torch.isnan(out)] = 0
        return out
    def forward(self,data1,data2):        
        res1=self.convert_rgb_hsv(data1)
        res2=self.convert_rgb_hsv(data2)
        loss = F.l1_loss(res1,res2)
        return loss

class LABLoss(nn.Module):
    def __init__(self):
        super(LABLoss,self).__init__()
    def forward(self,data1,data2):        
        rmean = ((data1[:,0] +data2[:,0] ) / 2)*255
        R = (data1[:,0] -data2[:,0])*255
        G = (data1[:,1] -data2[:,1])*255
        B = (data1[:,2] -data2[:,2])*255
        result = torch.mean((2+rmean/256)*(R**2)+4*(G**2)+(2+(255-rmean)/256)*(B**2))/(255**2)
        return result
