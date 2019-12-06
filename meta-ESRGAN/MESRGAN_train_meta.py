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
from MESRGAN_metalearner import MetaLearner
#from MESRGAN_metalearner import Net

from MESRGAN_discriminator import *

train_phase_name='meta-SGD'
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=140, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="div2k", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=70, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=23, help="number of residual blocks in generator")

#meta_optimizer.state_dict()
parser.add_argument("--saved_meta_optimizer_path",type=str,default=None,help="path of saved model")
#输入模型路径"saved_models/div2k/G_meta_0.pth"
parser.add_argument("--saved_model_path",type=str,default=None,help="path of saved model")

opt = parser.parse_args()

show_debug=True

cuda = torch.cuda.is_available()

model=MetaLearner(opt)
#will we need a discriminator?
#D=discriminator_VGG(channel_in=3,channel_gain=64,input_size=512)


#adam optimizer

# fetch loss function and metrics
Loss_per=nn.MSELoss()
Loss_L1=nn.L1Loss()
Loss_adv=nn.BCEWithLogitsLoss()

#currently unused--for evalution
#model_metrics = metrics

if cuda:
    model = model.cuda()
    D=D.cuda()
    Loss_per=Loss_per.cuda()
    Loss_L1=Loss_L1.cuda()
    Loss_adv=Loss_adv.cuda()
    VGG_ext=VGGFeatureExtractor(device=torch.device('cuda'))
    VGG_ext=VGG_ext.cuda()
else:
    VGG_ext=VGGFeatureExtractor(device=torch.device('cpu'))
model.define_task_lr_params()
model_params = list(model.parameters()) + list(model.task_lr.values())
meta_optimizer = torch.optim.Adam(model_params, lr=opt.lr,betas=(opt.b1, opt.b2))


if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("saved_models/%s/G_" %(opt.dataset_name)+train_phase_name+"_%d.pth" % (opt.epoch)))
else:
    # Initialize weights,here smaller sigma is better for trainning
    G.apply(weights_init)

dataloader =get_pic_dataloader("/"+opt.dataset_name,opt.batch_size,opt.n_cpu)
temp_save=work_folder+"/meta_SGD_temp"

def train_single_task(model, loss_fn, dataloaders, params):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.
    
    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of 
                     support set and query set
        params: (Params) hyperparameters
    """
    # set model to training mode
    model.train()

    # support set for a single few-shot task
    dl_sup = dataloaders['train']
    batch = dl_sup.__iter__().next()
    lr_img=batch["LR"]
    hr_img=batch["HR"]
    if params.cuda:
        lr_img, hr_img = lr_img.cuda(async=True), hr_img.cuda(async=True)

    # compute model output and loss
    fake_hr_img = model(lr_img)
    loss = loss_fn(fake_hr_img, hr_img)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    # NOTE if we want approx-MAML, change create_graph=True to False
    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # performs updates using calculated gradients
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        # NOTE Here Meta-SGD is different from naive MAML
        # Also we only need single update of inner gradient update
        task_lr = model.task_lr[key]
        adapted_params[key] = val - task_lr * grad
        adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict

def train_and_evaluate(model,
                       meta_optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file):
    """
    Train the model and evaluate every `save_summary_steps`.

    Args:
        model: (MetaLearner) a meta-learner for MAML algorithm
        meta_train_classes: (list) the classes for meta-training
        meta_val_classes: (list) the classes for meta-validating
        meta_test_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        meta_optimizer: (torch.optim) an meta-optimizer for MetaLearner
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:

        restore_path = os.path.join(args.model_dir,
                                    args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, meta_optimizer)

    # validation loss
    best_val_acc = -float('inf')

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Run inner loops to get adapted parameters (theta_t`)
            adapted_state_dicts = []
            dataloaders_list = []
            for n_task in range(params.num_inner_tasks):
                task = task_type(meta_train_classes, params.num_classes,
                                 params.num_samples, params.num_query)
                dataloaders = fetch_dataloaders(['train', 'test'], task)
                # Perform a gradient descent to meta-learner on the task
                a_dict = train_single_task(model, loss_fn, dataloaders, params)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(a_dict)
                dataloaders_list.append(dataloaders)

            # Update the parameters of meta-learner
            # Compute losses with adapted parameters along with corresponding tasks
            # Updated the parameters of meta-learner using sum of the losses
            meta_loss = 0
            for n_task in range(params.num_inner_tasks):
                dataloaders = dataloaders_list[n_task]
                dl_meta = dataloaders['test']  # query set
                X_meta, Y_meta = dl_meta.__iter__().next()
                if params.cuda:
                    X_meta, Y_meta = X_meta.cuda(async=True), Y_meta.cuda(
                        async=True)
                a_dict = adapted_state_dicts[n_task]
                Y_meta_hat = model(X_meta, a_dict)
                loss_t = loss_fn(Y_meta_hat, Y_meta)
                meta_loss += loss_t
            meta_loss /= float(params.num_inner_tasks)
            # print(meta_loss.item())

            # Meta-update using meta_optimizer
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            # print(model.task_lr.values())

            # Evaluate model on new task
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % params.save_summary_steps == 0:
                train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                         task_type, metrics, params, 'train')
                val_metrics = evaluate(model, loss_fn, meta_val_classes,
                                       task_type, metrics, params, 'val')
                test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                        task_type, metrics, params, 'test')

                train_loss = train_metrics['loss']
                val_loss = val_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                val_acc = val_metrics['accuracy']
                test_acc = test_metrics['accuracy']

                is_best = val_acc >= best_val_acc

                # Save weights
                utils.save_checkpoint(
                    {
                        'episode': episode + 1,
                        'state_dict': model.state_dict(),
                        'optim_dict': meta_optimizer.state_dict(),
                        'task_lr_dict': model.task_lr
                    },
                    is_best=is_best,
                    checkpoint=model_dir)

                # If best_test, best_save_path
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_val_acc = val_acc

                    # Save best test metrics in a json file in the model directory
                    best_train_json_path = os.path.join(
                        model_dir, "metrics_train_best_weights.json")
                    utils.save_dict_to_json(train_metrics,
                                            best_train_json_path)
                    best_val_json_path = os.path.join(
                        model_dir, "metrics_val_best_weights.json")
                    utils.save_dict_to_json(val_metrics, best_val_json_path)
                    best_test_json_path = os.path.join(
                        model_dir, "metrics_test_best_weights.json")
                    utils.save_dict_to_json(test_metrics, best_test_json_path)

                # Save latest test metrics in a json file in the model directory
                last_train_json_path = os.path.join(
                    model_dir, "metrics_train_last_weights.json")
                utils.save_dict_to_json(train_metrics, last_train_json_path)
                last_val_json_path = os.path.join(
                    model_dir, "metrics_val_last_weights.json")
                utils.save_dict_to_json(val_metrics, last_val_json_path)
                last_test_json_path = os.path.join(
                    model_dir, "metrics_test_last_weights.json")
                utils.save_dict_to_json(test_metrics, last_test_json_path)

                plot_history['train_loss'].append(train_loss)
                plot_history['train_acc'].append(train_acc)
                plot_history['val_loss'].append(val_loss)
                plot_history['val_acc'].append(val_acc)
                plot_history['test_loss'].append(test_loss)
                plot_history['test_acc'].append(test_acc)
                utils.plot_training_results(args.model_dir, plot_history)

                t.set_postfix(
                    tr_acc='{:05.3f}'.format(train_acc),
                    te_acc='{:05.3f}'.format(test_acc),
                    tr_loss='{:05.3f}'.format(train_loss),
                    te_loss='{:05.3f}'.format(test_loss))
                print('\n')

            t.update()

train_and_evaluate(model, meta_train_classes, meta_val_classes,
                    meta_test_classes, task_type, meta_optimizer, loss_fn,
                    model_metrics, params, args.model_dir,
                    args.restore_file)