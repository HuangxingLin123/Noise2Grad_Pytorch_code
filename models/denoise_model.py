import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import pytorch_ssim
import numpy as np



class DenoiseModel(BaseModel):
    def name(self):
        return 'DenoiseModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        parser.set_defaults(norm='batch', netG='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser



    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        if self.isTrain:

            self.visual_names = [ 'X','Y','X_denoise1','X_denoise2','n_hat','n_tilde','X_s','X_s_denoise']
        else:
            self.visual_names = ['X_denoise']

        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,id=2)
        self.netG = networks.define_G(opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.ssim_loss = pytorch_ssim.SSIM()

            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

    def gradient(self,img):
        img_5_h = img[:,:,:-1,:-1]
        img_6 = img[:,:,:-1,1:]
        h_res = img_6 - img_5_h

        img5_v = img[:,:,:-1,:-1]
        img_8 = img[:,:,1:,:-1]
        v_res = img_8 - img5_v

        grad =(h_res+v_res) * 0.5

        return grad


    def set_input(self, input,epoch,iteration):
        if self.isTrain:
            self.X = input['X'].to(self.device)
            self.Y = input['Y'].to(self.device)

            self.X_grad = self.gradient(self.X)

            self.epoch = epoch
            self.iteration = iteration

            self.image_paths = input['X_paths']
        else:
            self.X = input['X'].to(self.device)
            self.image_paths = input['X_paths']


    def forward(self):
        if self.isTrain:
            self.n_hat, self.n_tilde, self.X_denoise1, self.X_denoise2 = self.netG(self.X)

            self.n_grad = self.gradient(self.n_tilde)

            noise3 = self.n_tilde.detach()

            a = torch.ones_like(self.n_tilde) * 0.5
            mask = torch.bernoulli(a)
            mask = mask * 2 - 1

            self.X_s = noise3 * mask + self.Y

            self.X_s[self.X_s > 1.0] = 1.0
            self.X_s[self.X_s < 0] = 0
            _, _, self.X_s_denoise,_ = self.netG(self.X_s.detach())



        else:
            _,_ , self.X_denoise,_ = self.netG(self.X)



    def backward_G(self):
        tau = int(self.iteration / 500) + 1

        if self.iteration %tau == 0:
            self.loss_grad = self.criterionL2(self.n_grad, self.X_grad.detach())
        else:
            self.loss_grad = 0

        self.loss_Denoise = self.criterionL2(self.X_s_denoise, self.Y)



        self.loss_G = self.loss_Denoise + self.loss_grad



        self.loss_G.backward()




    def optimize_parameters(self):
        self.forward()
        # update D

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()