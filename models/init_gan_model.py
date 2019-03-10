import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class InitGANModel(BaseModel):
    def name(self):
        return 'InitGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default GAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['rec_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_HR_A', 'real_LR_A', 'fake_HR_A']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_B']

        # load/define networks
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_HR_A_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionRec = torch.nn.L1Loss()
            # self.criterionRec = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.real_HR_A = input['A'].to(self.device)
        # self.real_LR_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

        # load the ground-truth high resolution B image to test the SR quality
        # self.real_HR_B = input['GT_B'].to(self.device)
        # load the ground-truth low resolution A image
        self.real_LR_A = input['GT_A'].to(self.device)

    def forward(self):
        # LR -> HR
        # self.fake_HR_B = self.netG_B(self.real_LR_B)
        self.fake_HR_A = self.netG_B(self.real_LR_A)

    def psnr_eval(self):
        # compute the PSNR for the test
        self.bicubic_psnr = networks.compute_psnr(self.real_HR_A, self.real_LR_A)
        self.psnr = networks.compute_psnr(self.real_HR_A, self.fake_HR_A)

    def ssim_eval(self):
        self.bicubic_ssim = networks.compute_ssim(self.real_HR_A, self.real_LR_A)
        self.ssim = networks.compute_ssim(self.real_HR_A, self.fake_HR_A)

    def backward_G(self):
        # lambda_rec = self.opt.lambda_rec

        # reconstruct loss of high resolution fake_HR_A
        self.loss_rec_B = self.criterionRec(self.fake_HR_A, self.real_HR_A)
        self.loss_G = self.loss_rec_B

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
