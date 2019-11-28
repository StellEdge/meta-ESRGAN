import argparse
import torch
import torch.nn as nn
from psnr import PSNRLoss
from MESRGAN_generator import *
from dataloader import *
from Utils import *
from torch.autograd import Variable
import datetime
import time
import sys
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=140, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="div2k", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=70, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=12, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--saved_PSNR_model_path",type=str,default="saved_models/div2k/G_PSNR_30.pth",help="path of saved PSNR model")#输入PSNR模型路径
parser.add_argument("--n_residual_blocks", type=int, default=23, help="number of residual blocks in PSNR generator")
parser.add_argument("--saved_ESRGAN_model_path",type=str,default="saved_models/div2k/G_GAN_30.pth",help="path of saved ESRGAN model")#输入ESRGAN模型路径
parser.add_argument("--n_RRDB_blocks", type=int, default=23, help="number of residual blocks in ESRGAN generator")

opt = parser.parse_args()

cuda=torch.cuda.is_available()

G_PSNR=RRDBNet(3, 3, 64, opt.n_residual_blocks, gc=32)
G_GAN=RRDBNet(3, 3, 64, opt.n_RRDB_blocks, gc=32)
if cuda:
    print("Using CUDA.")
    G_PSNR=G_PSNR.cuda()
    G_GAN = G_GAN.cuda()

G_PSNR.load_state_dict(torch.load(opt.saved_PSNR_model_path))#加载已保存的PSNR模型
G_PSNR.eval()

G_GAN.load_state_dict(torch.load(opt.saved_ESRGAN_model_path))#加载已保存的ESRGAN模型
G_GAN.eval()

dataloader =get_pic_dataloader("/"+opt.dataset_name,opt.batch_size,opt.n_cpu)
temp_save=work_folder+"/temp"

def save_final_images(path):
    r_transform=transforms.Compose([transforms.ToPILImage()])
    for i, batch in enumerate(dataloader):
        lr_img=Variable(batch["LR"].cuda(),requires_grad=False)
        fake_PSNR=G_PSNR(lr_img)
        fake_GAN=G_GAN(lr_img)
        img_res=[r_transform(lr_img.data.cpu()[0]),  #origin image
                 r_transform(fake_PSNR.data.cpu()[0]),    #PSNR image 
                 r_transform(fake_GAN.data.cpu()[0]) #GAN image
                 ]
        for k,i,j in enumerate(img_res):
            alpha=0.0
            for m in range(11):#输出十一张图片，alpha取值0.0至1.0，每次递增0.1
                final_img=Image.blend(i,j,alpha)#(1-alpha)*PSNR+alpha*GAN
                final_img.save(os.path.join(temp_save,str(k)+'_'+str(alpha)+'.jpg'),quality=95)
                alpha+=0.1
        break

save_final_images(temp_save)