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
import matplotlib.pyplot as plt
import numpy as np

train_phase_name='PSNR'
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=20, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=140, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="div2k", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--start_up_epoch", type=int, default=1, help="epoch from which to end lr increase")
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=23, help="number of residual blocks in generator")

#6 res loss->0.05 ->GAN:DEAD

opt = parser.parse_args()

show_debug=True

cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

'''
Loss loss_G=loss_per+lamda*loss
'''
#psnr_loss=PSNRLoss()
loss_psnr=nn.L1Loss()
G=RRDBNet(3, 3, 64, opt.n_residual_blocks, gc=32) #origin 23 blocks



if cuda:
    print("Using CUDA.")
    G = G.cuda()
    loss_psnr=loss_psnr.cuda()


gen_params = list(G.parameters())
optimizer_G = torch.optim.Adam([p for p in gen_params if p.requires_grad],lr=opt.lr/opt.batch_size, betas=(opt.b1, opt.b2))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,opt.start_up_epoch, opt.decay_epoch).step
)


if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("saved_models/%s/G_" %(opt.dataset_name)+train_phase_name+"_%d.pth" % (opt.epoch)))
else:
    # Initialize weights,here smaller sigma is better for trainning
    G.apply(weights_init)

dataloader =get_pic_dataloader("/"+opt.dataset_name,opt.batch_size,opt.n_cpu)
temp_save=work_folder+"/psnr_temp"
def save_sample_images(path,label):
    G.eval()
    r_transform_n=transforms.Compose([
        transforms.Normalize(mean = [ -1, -1, -1 ],std = [ 2, 2, 2 ]),
        transforms.ToPILImage()
    ])
    for i, batch in enumerate(dataloader):
        lr_img=Variable(batch["LR"].cuda(),requires_grad=False)
        fake=G(lr_img)
        img_res=[r_transform_n(lr_img.data.cpu()[0]),  #origin image
                 r_transform_n(fake.data.cpu()[0])    #output image 
                 ]
        for k,i in enumerate(img_res):
            i.save(os.path.join(temp_save,label+'_'+str(k)+'.jpg'),quality=95)
        break
    G.train()

history_loss=[]
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    cur_epoch_loss=[]
    for i, batch in enumerate(dataloader):
        G.train()
        optimizer_G.zero_grad()
        input=Variable(batch["LR"].cuda(),requires_grad=True)
        ground_truth=Variable(batch["HR"].cuda(),requires_grad=False)
        #train G

        fake=G(input)
        #loss calculation
        loss_G=loss_psnr(fake,ground_truth)
        loss_G.backward()
        optimizer_G.step()
        #cur_epoch_loss.append(loss_G.item())
        history_loss.append(loss_G.item())
        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_G.item(),
                time_left,
            )
        )
        if batches_done % int(opt.sample_interval/opt.batch_size) == 0:
            pass
            #sample_transform(temp_save,str(epoch))
            save_sample_images(temp_save,str(epoch)+'_'+str(i))
    # Update learning rates
    #history_loss.append(torch.mean(Tensor(cur_epoch_loss)))
    los_avg = plt.subplot()
    los_avg.plot(history_loss, label="loss")
    plt.xlabel
    plt.xlabel("train iterations")
    plt.ylabel("loss")
    plt.title("losses per iteration")
    plt.savefig(temp_save + "/loss.jpg")
    plt.clf()
    plt.close()
    lr_scheduler_G.step()
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch!=0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/%s/G_" %(opt.dataset_name)+train_phase_name+"_%d.pth" % (epoch))