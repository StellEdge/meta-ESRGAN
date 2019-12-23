'''
discriminator for meta-ESRGAN
'''
import torch
import torch.nn as nn
import torchvision

#from xinntao
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=False,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            #ESRGAN hard coded value.
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class discriminator_VGG(nn.Module):
    def __init__(self,channel_in,channel_gain,input_size):
        super(discriminator_VGG,self).__init__()

        #first conv
        self.in_conv=[nn.Conv2d(channel_in,channel_gain,3,1,1,bias=True),
                      nn.LeakyReLU(0.2,True),
                      nn.Conv2d(channel_gain,channel_gain,3,2,1,bias=False),
                      nn.BatchNorm2d(channel_gain,affine=True),
                      nn.LeakyReLU(0.2,True),
                      ]
        self.in_conv=nn.Sequential(*self.in_conv)
        cur_dim=channel_gain
        cur_size=input_size/2
        self.conv_layers=[]
        for i in range(3):
            self.conv_layers+=self.build_conv_block(cur_dim)
            cur_dim*=2
            cur_size/=2
        self.conv_layers=nn.Sequential(*self.conv_layers)
        
        self.out_conv=[
            nn.Conv2d(cur_dim,cur_dim,3,1,1,bias=False),
            nn.BatchNorm2d(cur_dim,affine=True),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(cur_dim,cur_dim,3,2,1,bias=False),
            nn.BatchNorm2d(cur_dim,affine=True),
            nn.LeakyReLU(0.2,True),
            ]
        '''
        self.out_conv=[
            nn.Conv2d(cur_dim,1,3,1,1,bias=True),
            ]
        '''
        self.out_conv=nn.Sequential(*self.out_conv)
        
        cur_size/=2
        cur_size=int(cur_size)
        self.linear =[nn.Linear(cur_dim * cur_size * cur_size, 100),
                      nn.LeakyReLU(0.2,True),
                      nn.Linear(100, 1)
                      ]
        self.linear=nn.Sequential(*self.linear)
        
    def build_conv_block(self,channel_gain):
        model=[
            nn.Conv2d(channel_gain,channel_gain*2,3,1,1,bias=True),
            nn.BatchNorm2d(channel_gain*2,affine=True),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(channel_gain*2,channel_gain*2,4,2,1,bias=True),
            nn.BatchNorm2d(channel_gain*2,affine=True),
            nn.LeakyReLU(0.2,True),
            ]
        return model
    def forward(self,x):
        x=self.in_conv(x)
        x=self.conv_layers(x)
        x=self.out_conv(x)
        x=x.view(x.size(0),-1)
        out=self.linear(x)
        return out