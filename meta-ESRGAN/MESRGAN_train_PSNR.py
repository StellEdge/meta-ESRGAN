import argparse
import torch
import torch.nn as nn
from psnr import PSNRLoss
from MESRGAN_generator import *
from dataloader import *
from Utils import *



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

dataloader =get_pic_dataloader("/"+opt.dataset_name,opt.batch_size)

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        print(batch["LR"].size())
        print(batch["HR"].size())
        if cuda:
            input=Variable(batch["X"].cuda(),requires_grad=False)
            ground_truth=Variable(batch["Y"].cuda(),requires_grad=False)
        else:
            input=Variable(batch["X"],requires_grad=False)
            ground_truth=Variable(batch["Y"],requires_grad=False)
        #train G
        G.train()
        optimizer_G.zero_grad()
        fake=G(input)
        #loss calculation
        loss_G=loss_psnr(fake,ground_truth)
        loss_G.backward()
        optimizer_G.step()

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
        if batches_done % opt.sample_interval == 0:
            pass
            #sample_transform(temp_save,str(epoch))
            #save_sample_images(temp_save,str(epoch)+'_'+str(i))
    # Update learning rates

    lr_scheduler_G.step()
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch!=0:
        # Save model checkpoints
        torch.save(G.state_dict(), "saved_models/%s/G_"+train_phase_name+"_%d.pth" % (opt.dataset_name, epoch))