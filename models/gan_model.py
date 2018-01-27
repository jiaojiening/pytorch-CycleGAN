import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class GANModel(BaseModel):
    def name(self):
        return 'GANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default GAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_Rec', type=float, default=10.0, help='weight for reconstruction loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_B', 'G_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'GT_A']
        visual_names_B = ['real_B', 'fake_A', 'GT_B']

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_B', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_B']

        # load/define networks
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_HR_A_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionRec = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain( self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_HR_A = input['A'].to(self.device)
        self.real_LR_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

        # load the ground-truth high resolution B image to test the SR quality
        self.real_HR_B = input['GT_B'].to(self.device)
        # load the ground-truth low resolution A image
        self.real_LR_A = input['GT_A'].to(self.device)

    def forward(self):
        # LR -> HR
        self.fake_HR_B = self.netG_B(self.real_LR_B)
        self.fake_HR_A = self.netG_B(self.real_LR_A)

    def psnr_eval(self):
        # compute the PSNR for the test
        self.bicubic_psnr = networks.compute_psnr(self.real_HR_B, self.real_LR_B)
        self.psnr = networks.compute_psnr(self.real_HR_B, self.fake_HR_B)

    def ssim_eval(self):
        self.bicubic_ssim = networks.compute_ssim(self.real_HR_B, self.real_LR_B)
        self.ssim = networks.compute_ssim(self.real_HR_B, self.fake_HR_B)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        # fake.detach() the loss_D do not backward to the net_G
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_B(self):
        fake_HR_A = self.fake_HR_A_pool.query(self.fake_HR_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_HR_A, fake_HR_A)

    def backward_G(self):
        lambda_Rec = self.opt.lambda_Rec

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_HR_A), True)
        # combined loss
        self.loss_G = self.loss_G_B

        # reconstruct loss of high resolution fake_HR_A
        self.loss_rec = self.criterionRec(self.fake_HR_A, self.real_HR_A) * lambda_Rec
        self.loss_G += self.loss_rec

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_B
        self.set_requires_grad([self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B
        self.set_requires_grad([self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_B()
        self.optimizer_D.step()
