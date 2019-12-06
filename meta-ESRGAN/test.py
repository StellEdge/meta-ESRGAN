from MESRGAN_RRDB_meta import *
import argparse
import torch
import torch.nn as nn
from psnr import PSNRLoss
from dataloader import *
from Utils import *
from torch.autograd import Variable
import datetime
import time
import sys
from PIL import Image

G=RRDBNet_meta(3, 3, 64, 23, gc=32)

G.load_state_dict(torch.load("../model/G_PSNR_20.pth",map_location=torch.device('cpu')))
G.eval()

'''
print("the state_dict of G:")
for i in G.state_dict():
    print(i, "\t", G.state_dict()[i].size())
'''

#print(G.named_parameters())

dataset=get_pic_dataset("")
temp_save="../temp"

def save_final_images(path):
    with torch.no_grad():
        r_transform=transforms.Compose([
            transforms.Normalize(mean = [ -1, -1, -1 ],std = [ 2, 2, 2 ]),
            transforms.ToPILImage()
        ])
        for k in range(10):
            batch=dataset.__getitem__(k)
            lr_img=batch["LR"].unsqueeze(0).contiguous()
            fake_PSNR=G(lr_img,OrderedDict(G.named_parameters()))
            fake_origin=G(lr_img)
            
            img_res=[r_transform(lr_img.data.cpu()[0]),  #origin image
                        r_transform(fake_PSNR.data.cpu()[0]),    #PSNR image
                        r_transform(fake_origin.data.cpu()[0])
                        ]
            for i in range(len(img_res)):
                img_res[i].save(os.path.join(temp_save,str(k)+'output_base_'+str(i)+'.jpg'),quality=100)
            


save_final_images(temp_save)