import torch


class LambdaLR:
    #OFFERING decay-learning rate
    def __init__(self, n_epochs, offset, start_up_epoch,decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        self.start_up_epoch = start_up_epoch
    def step(self, epoch):
        if epoch<self.start_up_epoch:
            return 0.5+0.5*max(0,epoch/ self.start_up_epoch)
        else:
            return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.2, mode='fan_in',nonlinearity='leaky_relu')
        #here 0.2 refers to the leakyReLU init
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data,0.2, mode='fan_in',nonlinearity='leaky_relu')
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)
