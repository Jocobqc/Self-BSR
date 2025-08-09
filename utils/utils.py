from torch.nn import init
from torch.optim import lr_scheduler
import os
import torchvision.utils as utils
import torch
import numpy as np
import random
import cv2

def save_checkpoint(model, epoch, checkpoints_dir,save_type="normal"):
    if save_type == "best_psnr":
        model_name = 'epoch_best_psnr.pth'
    elif save_type == "best_ssim":
        model_name = 'epoch_best_ssim.pth'
    else:
        model_name = 'epoch_{epoch}.pth'.format(epoch=epoch)
    path = os.path.join(checkpoints_dir,  model_name)
    torch.save(model.state_dict(), path)

def img_save(img, img_path):
    cv2.imwrite(img_path, tensor2uint(img))

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_match_dict(model, model_path):
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with [%s]' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_scheduler(opt,optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - int(opt.epochs/2)) / float(int(opt.epochs/2)+1 )
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(opt.epochs/8), eta_min=1e-6)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy) 
    print('learning rate policy [%s] is implemented'%opt.lr_policy)   
    return scheduler

def save_mutil_imgs(val_images,training_results_path,epoch,row_num=3, img_num=1):
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // int(len(val_images)/img_num))
    index = 1
    for i in range(len(val_images)):
        images = val_images[i]        
        image = utils.make_grid(images, nrow=row_num, padding=5)
        utils.save_image(image, training_results_path + '/epoch_%d_index_%d.png' % (epoch, index), padding=5)
        index += 1

def setup_seed(opt):
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)