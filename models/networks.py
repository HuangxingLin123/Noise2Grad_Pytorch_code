import torch
import torch.nn as nn
from torch.nn import init
from .unet_parts import *
import functools
from torch.optim import lr_scheduler



###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = denoisenet(feature_num=32)


    return init_net(net, init_type, init_gain, gpu_ids)



class denoisenet(nn.Module):
    def __init__(self, feature_num=8):
        super(denoisenet, self).__init__()

        self.inc = inconv(3, feature_num)

        self.down1 = down(feature_num, feature_num*2)
        self.down2 = down(feature_num*2, feature_num*4)
        self.down3 = down(feature_num*4, feature_num*4)

        self.up1 = up(feature_num*8, feature_num*2)
        self.up2 = up(feature_num*4, feature_num*1)
        self.up3 = up(feature_num*2, feature_num)

        self.outc=nn.Sequential(
            nn.Conv2d(feature_num, 3, kernel_size=3, stride=1, padding=1),

        )


       # Noise approximation module
        self.s = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
        )



    def forward(self, input1):

        fi = self.inc(input1)
        x2 = self.down1(fi)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        ff3 = self.up3(x, fi)

        cont1 = self.outc(ff3)

        n_hat = input1 - cont1
        n_tilde = self.s(n_hat)

        cont2 = input1 - n_tilde

        return n_hat, n_tilde, cont1, cont2





