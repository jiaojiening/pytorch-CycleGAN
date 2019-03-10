import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import networks_reid


class ReidHybridCycleGANModel(BaseModel):
    def name(self):
        return 'ReidHybridCycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default GAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. '
                                     'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_G', type=float, default=1.0, help='weight for Generator loss')
            # reid parameters
            parser.add_argument('--droprate', type=float, default=0.5, help='the dropout ratio in reid model')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'rec_A', 'rec_B', 'reid']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'reid']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_HR_A', 'fake_LR_A', 'rec_HR_A', 'real_LR_A']
        visual_names_B = ['real_LR_B', 'fake_HR_B', 'rec_LR_B', 'real_HR_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_reid']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'D_reid']

        # netG_A: HR -> LR, netG_B: LR -> HR
        # load/define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Load a pretrained resnet model and reset the final connected layer
        self.netD_reid = networks_reid.ft_net(opt.num_classes, opt.droprate)
        # the reid network is trained on a single gpu because of the BatchNorm layer
        self.netD_reid = self.netD_reid.to(self.device)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                            self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                            self.gpu_ids)

        if self.isTrain:
            # GAN
            self.fake_HR_A_pool = ImagePool(opt.pool_size)
            # CycleGAN
            self.fake_LR_A_pool = ImagePool(opt.pool_size)  # fake_B_pool
            self.fake_HR_B_pool = ImagePool(opt.pool_size)  # fake_A_pool
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionRec = torch.nn.L1Loss()
            self.criterionReid = torch.nn.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            # SR optimizer
            # self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # reid optimizer
            ignored_params = list(map(id, self.netD_reid.model.fc.parameters())) + \
                             list(map(id, self.netD_reid.classifier.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.netD_reid.parameters())
            self.optimizer_D_reid = torch.optim.SGD([
                {'params': base_params, 'lr': 0.1 * opt.reid_lr},
                {'params': self.netD_reid.model.fc.parameters(), 'lr': opt.reid_lr},
                {'params': self.netD_reid.classifier.parameters(), 'lr': opt.reid_lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

            self.optimizer_reid.append(self.optimizer_D_reid)

    def reset_model_status(self):
        if self.opt.stage==1:
            self.netG_A.train()
            self.netG_B.train()
            self.netD_A.train()
            self.netD_B.train()
            # for the BatchNorm
            self.netD_reid.eval()
        elif self.opt.stage==0 or self.opt.stage==2:
            self.netG_A.train()
            self.netG_B.train()
            self.netD_A.train()
            self.netD_B.train()
            # for the BatchNorm
            self.netD_reid.train()

    def set_input(self, input):
        self.real_HR_A = input['A'].to(self.device)
        self.real_LR_B = input['B'].to(self.device)
        # load the ground-truth low resolution A image
        self.real_LR_A = input['GT_A'].to(self.device)

        # load the ground-truth high resolution B image to test the SR quality
        self.real_HR_B = input['GT_B'].to(self.device)

        self.image_paths = input['A_paths']

        # get the id label for person reid
        self.A_label = input['A_label'].to(self.device)
        self.B_label = input['B_label'].to(self.device)

    def forward(self):
        # GAN
        self.fake_HR_A = self.netG_B(self.real_LR_A)  # LR -> HR
        # cycleGAN
        # HR -> LR -> HR
        self.fake_LR_A = self.netG_A(self.real_HR_A)  # HR -> LR
        self.rec_HR_A = self.netG_B(self.fake_LR_A)  # LR -> HR
        # LR -> HR -> LR
        self.fake_HR_B = self.netG_B(self.real_LR_B)  # LR -> HR
        self.rec_LR_B = self.netG_A(self.fake_HR_B)   # HR -> LR

        # self.imags = torch.cat([self.real_HR_A, self.fake_HR_B], 0)
        # self.labels = torch.cat([self.A_label, self.B_label], 0)
        # all the HR images
        self.imgs = torch.cat([self.real_HR_A, self.fake_HR_B, self.rec_HR_A, self.fake_HR_A], 0)
        self.labels = torch.cat([self.A_label, self.B_label, self.A_label, self.A_label])
        self.pred_imgs = self.netD_reid(self.imgs)

    def psnr_eval(self):
        # compute the PSNR for the test
        self.bicubic_psnr = networks.compute_psnr(self.real_HR_A, self.real_LR_A)
        self.psnr = networks.compute_psnr(self.real_HR_A, self.fake_HR_A)

    def ssim_eval(self):
        self.bicubic_ssim = networks.compute_ssim(self.real_HR_A, self.real_LR_A)
        self.ssim = networks.compute_ssim(self.real_HR_A, self.fake_HR_A)

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

    def backward_D_A(self):
        # real/fake LR image(G_A)
        fake_LR_A = self.fake_LR_A_pool.query(self.fake_LR_A)
        # # used for GAN
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_LR_A, fake_LR_A)
        # # used for CycleGAN
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_LR_B, fake_LR_A)
        real_LR = torch.cat([self.real_LR_A, self.real_LR_B], 0)
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_LR, fake_LR_A)

    def backward_D_B(self):
        fake_HR_A = self.fake_HR_A_pool.query(self.fake_HR_A)  # GAN
        fake_HR_B = self.fake_HR_B_pool.query(self.fake_HR_B)
        # # used for GAN
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_HR_A, fake_HR_A)
        # # used for CycleGAN
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_HR_A, fake_HR_B)
        fake_HR = torch.cat([fake_HR_A, fake_HR_B], 0)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_HR_A, fake_HR)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_rec = self.opt.lambda_rec
        lambda_G = self.opt.lambda_G

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_LR_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_LR_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_HR_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_HR_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_LR_A), True)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_LR_A), True) * lambda_G
        # GAN loss D_B(G_B(B))
        # used for GAN
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_HR_A), True)
        # used for CycleGAN
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_HR_B), True)
        fake_HR = torch.cat([self.fake_HR_A, self.fake_HR_B], 0)
        # self.loss_G_B = self.criterionGAN(self.netD_B(fake_HR), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(fake_HR), True) * lambda_G
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_HR_A, self.real_HR_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_LR_B, self.real_LR_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        # reconstruct loss of low resolution fake_LR_A(G_A)
        self.loss_rec_A = self.criterionRec(self.fake_LR_A, self.real_LR_A) * lambda_rec
        # reconstruct loss of high resolution fake_HR_A(G_B)
        self.loss_rec_B = self.criterionRec(self.fake_HR_A, self.real_HR_A) * lambda_rec
        self.loss_rec = self.loss_rec_A + self.loss_rec_B

        self.loss_G += self.loss_rec

        _, pred_label_imgs = torch.max(self.pred_imgs, 1)
        self.corrects += float(torch.sum(pred_label_imgs == self.labels))
        self.loss_reid = self.criterionReid(self.pred_imgs, self.labels)

        self.loss_G = self.loss_G + self.loss_reid

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        if self.opt.stage == 1:
            # G_A and G_B
            # self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_reid], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
        if self.opt.stage == 0 or self.opt.stage == 2:
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.optimizer_G.zero_grad()
            self.optimizer_D_reid.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            self.optimizer_D_reid.step()
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()