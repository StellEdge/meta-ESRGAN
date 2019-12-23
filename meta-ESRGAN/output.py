import argparse
import torch
import torch.nn as nn
from psnr import PSNRLoss
from MESRGAN_generator import *
from MESRGAN_RRDB_meta import *
from dataloader import *
from Utils import *
from torch.autograd import Variable
import datetime
import time
import sys
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="div2k", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--decay_epoch", type=int, default=70, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--saved_PSNR_model_path",type=str,default="saved_models/div2k/G_HSV_60.pth",help="path of saved PSNR model")#输入PSNR模型路径
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in PSNR generator")
parser.add_argument("--saved_ESRGAN_model_path",type=str,default="saved_models/div2k/G_GAN_155.pth",help="path of saved ESRGAN model")#输入ESRGAN模型路径
parser.add_argument("--n_RRDB_blocks", type=int, default=23, help="number of residual blocks in ESRGAN generator")

opt = parser.parse_args()

cuda=torch.cuda.is_available()

G_PSNR=RRDBNet_shuffle(3, 3, 64, opt.n_residual_blocks, gc=32)
G_PSNR_meta=RRDBNet_shuffle(3, 3, 64, opt.n_residual_blocks, gc=32)

G_GAN=RRDBNet_shuffle(3, 3, 64, opt.n_residual_blocks, gc=32)
if cuda:
    print("Using CUDA.")
    G_PSNR=G_PSNR.cuda()
    G_PSNR_meta=G_PSNR_meta.cuda()
    G_GAN = G_GAN.cuda()

G_PSNR.load_state_dict(torch.load(opt.saved_PSNR_model_path))#加载已保存的PSNR模型
G_PSNR.eval()

#G_PSNR_meta.load_state_dict(torch.load(opt.saved_PSNR_model_path))#加载已保存的PSNR模型
G_PSNR_meta.eval()

G_GAN.load_state_dict(torch.load(opt.saved_ESRGAN_model_path))#加载已保存的ESRGAN模型
G_GAN.eval()

new_params=[OrderedDict() for i in range(4)]
for x, y in zip(list(G_GAN.named_parameters()), list(G_PSNR.named_parameters())):
    prefix=x[0]
    alpha=0.2
    for i in range(4):
        new_params[i][prefix]=alpha*x[1]+(1-alpha)*y[1]
        alpha+=0.2

#dataloader =get_pic_dataloader("/"+opt.dataset_name,opt.batch_size,opt.n_cpu)
dataset=get_pic_dataset("/"+opt.dataset_name+'_test')
temp_save=work_folder+"/temp"

def tensorclamp(t):
    r=torch.clamp(t,0,1)
    return r

def save_final_images(path):
    with torch.no_grad():
        r_transform=transforms.Compose([
            #transforms.Normalize(mean = [ -1, -1, -1 ],std = [ 2, 2, 2 ]),
            transforms.Lambda(tensorclamp),
            transforms.ToPILImage()
        ])
        for k in range(50):
            batch=dataset.__getitem__(k)
            lr_img=batch["LR"].cuda().unsqueeze(0).contiguous()
            hr_img=batch["HR"]
            fake_PSNR=G_PSNR(lr_img) 

            #fake_PSNR_meta=G_PSNR_meta(lr_img)
            G_PSNR_meta.load_state_dict(new_params[0])
            fake_PSNR_meta_0=G_PSNR_meta(lr_img)

            G_PSNR_meta.load_state_dict(new_params[1])
            fake_PSNR_meta_1=G_PSNR_meta(lr_img)

            G_PSNR_meta.load_state_dict(new_params[2])
            fake_PSNR_meta_2=G_PSNR_meta(lr_img)

            G_PSNR_meta.load_state_dict(new_params[3])
            fake_PSNR_meta_3=G_PSNR_meta(lr_img)

            fake_GAN=G_GAN(lr_img)
            
            img_res=[r_transform(lr_img.data.cpu()[0]),  #origin image
                        r_transform(fake_PSNR.data.cpu()[0]),    #PSNR image
                        r_transform(fake_PSNR_meta_0.data.cpu()[0]),    #PSNR image
                        r_transform(fake_PSNR_meta_1.data.cpu()[0]),    #PSNR image
                        r_transform(fake_PSNR_meta_2.data.cpu()[0]),    #PSNR image
                        r_transform(fake_PSNR_meta_3.data.cpu()[0]),    #PSNR image
                        r_transform(fake_GAN.data.cpu()[0]), #GAN image
                        r_transform(hr_img)
                        ]
            for i in range(len(img_res)):
                img_res[i].save(os.path.join(temp_save,str(k)+'output_base_'+str(i)+'.jpg'),quality=100)
            alpha=0.0
            '''
            for m in range(10):#输出十一张图片，alpha取值0.0至1.0，每次递增0.1
                final_img=Image.blend(img_res[1],img_res[3],alpha)#(1-alpha)*PSNR+alpha*GAN
                final_img.save(os.path.join(temp_save,str(k)+'output_alpha_'+str(alpha)+'.jpg'),quality=100)
                alpha+=0.1
            '''

save_final_images(temp_save)