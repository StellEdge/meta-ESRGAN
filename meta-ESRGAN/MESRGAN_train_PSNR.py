import argparse
import torch
import torch.nn as nn
from psnr import PSNRLoss
from MESRGAN_generator import *
from dataloader import *
from Utils import *
import matplotlib.pyplot as plt
train_phase_name='PSNR'
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="div2k", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=400, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10, help="cycle loss weight")
parser.add_argument("--lambda_cha", type=float, default=1.0, help="characteristic loss weight")
opt = parser.parse_args()

show_debug=True

cuda = torch.cuda.is_available()

#Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

'''
Loss loss_G=loss_per+lamda*loss
'''
#psnr_loss=PSNRLoss()
loss_psnr=nn.L1Loss()
G=RRDBNet(3, 3, 64, 23, gc=32)



if cuda:
    print("Using CUDA.")
    G = G.cuda()
    loss_psnr=loss_psnr.cuda()

train_dataset=get_pic_dataset("/"+opt.dataset_name)

gen_params = list(G.parameters())
optimizer_G = torch.optim.Adam([p for p in gen_params if p.requires_grad],lr=opt.lr, betas=(opt.b1, opt.b2))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)


if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("saved_models/%s/G_"+train_phase_name+"_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights,here smaller sigma is better for trainning
    G.apply(weights_init)

print(train_dataset.class_to_idx)
plt.imshow(train_dataset[0][0])
plt.axis('off')
plt.show()

prev_time = time.time()
#for epoch in range(opt.epoch, opt.n_epochs):