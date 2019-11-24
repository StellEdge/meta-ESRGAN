import argparse
import torch
import torch.nn as nn
from psnr import PSNRLoss
from MESRGAN_generator import *
from MESRGAN_discriminator import *
from dataloader import *
from Utils import *
from torch.autograd import Variable
import datetime
import time
import sys

train_phase_name='GAN'
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--n_PSNR_epochs", type=int, default=60, help="number of PSNR epochs of training")
parser.add_argument("--dataset_name", type=str, default="div2k", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_RRDB_blocks", type=int, default=23, help="number of residual blocks in generator")
parser.add_argument("--lambda_gan", type=float, default=10, help="gan loss weight")
parser.add_argument("--lambda_L1", type=float, default=1.0, help="L1 loss weight")
opt = parser.parse_args()

show_debug=True

cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#loss_G=Loss_per+lambda_gan*Loss_Ra_D+lambda_L1*Loss_L1

Loss_per=nn.MSELoss()
Loss_L1=nn.L1Loss()
Loss_adv=nn.BCEWithLogitsLoss()
G=RRDBNet(3, 3, 64, opt.n_RRDB_blocks, gc=32)
D=discriminator_VGG(channel_in=3,channel_gain=64,input_size=512)

train_dataset=get_pic_dataset("/"+opt.dataset_name)

if cuda:
    print("Using CUDA.")
    G = G.cuda()
    D = D.cuda()
    Loss_per=Loss_per.cuda()
    Loss_L1=Loss_L1.cuda()
    Loss_adv=Loss_adv.cuda()
    VGG_ext=VGGFeatureExtractor(device=torch.device('cuda'))
    VGG_ext=VGG_ext.cuda()
else:
    VGG_ext=VGGFeatureExtractor(device=torch.device('cpu'))


gen_params = list(G.parameters())
optimizer_G = torch.optim.Adam([p for p in gen_params if p.requires_grad],lr=opt.lr, betas=(opt.b1, opt.b2))
dis_params = list(D.parameters())
optimizer_D= torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=opt.lr, betas=(opt.b1, opt.b2))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)


if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("saved_models/%s/G_" % (opt.dataset_name)+train_phase_name+"_%d.pth" % (opt.epoch)))
    D.load_state_dict(torch.load("saved_models/%s/D_" % (opt.dataset_name)+train_phase_name+"_%d.pth" (opt.epoch)))
else:
    G.load_state_dict(torch.load("saved_models/%s/G_PSNR_%d.pth" % (opt.dataset_name, opt.n_PSNR_epochs)))
    D.apply(weights_init)

dataloader =get_pic_dataloader("/"+opt.dataset_name,opt.batch_size)
temp_save=work_folder+"/gan_temp"
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


prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        input=Variable(batch["LR"].cuda())
        ground_truth=Variable(batch["HR"].cuda())
        #train G
        G.train()
        optimizer_G.zero_grad()
        fake_HR=G(input)
        #loss calculation
        loss_per=Loss_per(VGG_ext(ground_truth),VGG_ext(fake_HR))

        dis_real=D(ground_truth.detach())
        dis_fake=D(fake_HR)

        valid = Variable( torch.ones_like(dis_real.data).cuda(),  requires_grad=False)
        fake = Variable( torch.zeros_like(dis_real.data).cuda(), requires_grad=False)   

        loss_Ra_D=Loss_adv(dis_fake - dis_real.mean(0, keepdim=True), valid)
        loss_L1=Loss_L1(ground_truth,fake_HR)
        loss_G=loss_per+opt.lambda_gan*loss_Ra_D+opt.lambda_L1*loss_L1

        loss_G.backward()
        optimizer_G.step()

        #train D
        D.train()
        optimizer_D.zero_grad()
        dis_real=D(ground_truth)
        dis_fake=D(fake_HR.detach())

        loss_real = Loss_adv(dis_real - dis_fake.mean(0, keepdim=True), valid)
        loss_fake = Loss_adv(dis_fake - dis_real.mean(0, keepdim=True), fake)

        loss_D=(loss_real+loss_fake)/2
        loss_D.backward()
        optimizer_D.step()
        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f  loss per:%f loss adv:%f loss L1:%f] [D loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_G.item(),
                loss_per.item(),
                loss_Ra_D.item(),
                loss_L1.item(),
                loss_D.item(),
                time_left,
            )
        )
        if batches_done % opt.sample_interval == 0:
            #sample_transform(temp_save,str(epoch))
            save_sample_images(temp_save,str(epoch)+'_'+str(i))
    # Update learning rates

    lr_scheduler_G.step()
    lr_scheduler_D.step()
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch!=0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/%s/G_"%(opt.dataset_name)+train_phase_name+"_%d.pth" %(epoch))
        torch.save(D.state_dict(), "saved_models/%s/D_"%(opt.dataset_name)+train_phase_name+"_%d.pth" %(epoch))


