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

def weights_init_discriminator(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.2, mode='fan_in',nonlinearity='leaky_relu')
        #here 0.2 refers to the leakyReLU init
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data,0,0.1)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".
              format(checkpoint))
        os.mkdir(checkpoint)
    else:
        # print("Checkpoint Directory exists! ")
        pass
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.task_lr = checkpoint['task_lr_dict']

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


