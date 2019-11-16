import torch
import torch.nn as nn

def PSNR_Loss(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class PSNRLoss(nn.module):
    def __init__(self):
        super(PSNRLoss,self).__init__()
    def forward(self,x,y):
        mse=torch.mean((x/255.-y/255.)**2)
        PIXEL_MAX = 1
        return 20 * torch.log10(PIXEL_MAX /torch.sqrt(mse))

