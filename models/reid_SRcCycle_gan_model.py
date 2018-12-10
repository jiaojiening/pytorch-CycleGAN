import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import network_reid


class ReidSRcCycleGANModel(BaseModel):
    def name(self):
        return 'ReidSRcCycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            # reid parameters
            # parser.add_argument('--num_classes', type=int, default=702, help='The total num of the id classes ')
            parser.add_argument('--droprate', type=float, default=0.5, help='the dropout ratio in reid model')
            parser.add_argument('--reid_lr', type=float, default=0.1, help='initial reid learning rate for adam')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'reid']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'GT_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_reid']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'D_reid']

        self.num_attr = opt.num_attr

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # Input: B(low-resolution) + attributes of A(high-resolution)
        self.netG_B = networks.define_G(opt.output_nc + opt.num_attr, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Load a pretrained resnet model and reset the final connected layer
        self.netD_reid = network_reid.ft_net(opt.num_classes, opt.droprate)
        # the reid network is trained on a single gpu because of the BatchNorm layer
        self.netD_reid = self.netD_reid.to(self.device)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            # Input1: A(high-resolution) + attribute of A(high-resolution)
            # Input2: fake_A + attribute of A(high-resolution)
            # Input3: A(high-resolution) + random sampled attribute
            condit_netD = 'conditional_basic'
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, condit_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids,
                                            opt.num_attr)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionReid = torch.nn.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # reid optimizer
            ignored_params = list(map(id, self.netD_reid.model.fc.parameters())) + \
                             list(map(id, self.netD_reid.classifier.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.netD_reid.parameters())
            self.optimizer_D_reid = torch.optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.reid_lr},
                {'params': self.netD_reid.model.fc.parameters(), 'lr': opt.reid_lr},
                {'params': self.netD_reid.classifier.parameters(), 'lr': opt.reid_lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

            # Decay learning rate by a factor of 0.1 every 40 epochs
            self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D_reid,
                                                                    step_size=40, gamma=0.1)
            # self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D_reid,
            #                                                         step_size=20, gamma=0.1)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # add the conditional attributes vector
        self.A_real_attr = input['A_real_attr'].to(self.device)
        self.A_fake_attr = input['A_fake_attr'].to(self.device)
        self.B_real_attr = input['B_real_attr'].to(self.device)

        # load the ground-truth high resolution B image to test the SR quality
        # if not self.isTrain:
        self.GT_B = input['GT_B'].to(self.device)

        # get the id label for person reid
        self.A_label = input['A_label'].to(self.device)
        self.B_label = input['B_label'].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)

        # print(self.fake_B.size()) # [1, 3, 256, 128]
        A_real_attr = torch.unsqueeze(self.A_real_attr, 2)  # add a new axis
        # A_real_attr = A_real_attr.repeat(1, 1, 128 * 128)
        # A_real_attr = torch.reshape(A_real_attr, (-1, self.num_attr, 128, 128))
        A_real_attr = A_real_attr.repeat(1, 1, self.fake_B.size()[2] * self.fake_B.size()[3])
        A_real_attr = torch.reshape(A_real_attr, (-1, self.num_attr, self.fake_B.size()[2], self.fake_B.size()[3]))
        A_real_attr = A_real_attr.float()
        comb_input_fake = torch.cat([A_real_attr, self.fake_B], 1)
        self.rec_A = self.netG_B(comb_input_fake)

        # print(self.real_B.size()) # [1, 3, 256, 128]
        # self.fake_A = self.netG_B(self.real_B)
        B_real_attr = torch.unsqueeze(self.B_real_attr, 2)  # add a new axis
        # B_real_attr = B_real_attr.repeat(1, 1, 128 * 128)
        # B_real_attr = torch.reshape(B_real_attr, (-1, self.num_attr, 128, 128))
        B_real_attr = B_real_attr.repeat(1, 1, self.real_B.size()[2] * self.real_B.size()[3])
        B_real_attr = torch.reshape(B_real_attr, (-1, self.num_attr, self.real_B.size()[2], self.real_B.size()[3]))
        B_real_attr = B_real_attr.float()

        comb_input_real = torch.cat([B_real_attr, self.real_B], 1)
        self.fake_A = self.netG_B(comb_input_real)
        self.rec_B = self.netG_A(self.fake_A)

        # person re-id prediction of HR images
        # more strong than the D_B loss
        self.netD_reid = self.netD_reid.train()
        self.pred_real_A = self.netD_reid(self.real_A)   # A_label HR
        self.pred_fake_A = self.netD_reid(self.fake_A)   # B_label HR

    def psnr_eval(self):
        # compute the PSNR for the test
        self.bicubic_psnr = networks.compute_psnr(self.GT_B, self.real_B)
        self.psnr = networks.compute_psnr(self.GT_B, self.fake_A)

    def ssim_eval(self):
        self.bicubic_ssim = networks.compute_ssim(self.GT_B, self.real_B)
        self.ssim = networks.compute_ssim(self.GT_B, self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_condit(self, netD, real, fake, real_attr, fake_attr):
        # Real
        pred_real = netD(real, real_attr)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake_1 = netD(fake.detach(), real_attr)
        loss_D_fake_1 = self.criterionGAN(pred_fake_1, False)
        pred_fake_2 = netD(real, fake_attr)
        loss_D_fake_2 = self.criterionGAN(pred_fake_2, False)
        loss_D_fake = (loss_D_fake_1 + loss_D_fake_2) * 0.5

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_condit(self.netD_B, self.real_A, fake_A, self.A_real_attr, self.B_real_attr)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.

            # print(self.real_A.size())   # [1, 3, 256, 128]
            A_real_attr = torch.unsqueeze(self.A_real_attr, 2)  # add a new axis
            # A_real_attr = A_real_attr.repeat(1, 1, 128 * 128)
            # A_real_attr = torch.reshape(A_real_attr, (-1, self.num_attr, 128, 128))
            A_real_attr = A_real_attr.repeat(1, 1, self.real_A.size()[2] * self.real_A.size()[3])
            A_real_attr = torch.reshape(A_real_attr, (-1, self.num_attr, self.real_A.size()[2], self.real_A.size()[3]))
            A_real_attr = A_real_attr.float()
            self.comb_real_A = torch.cat([A_real_attr, self.real_A], 1)

            self.idt_B = self.netG_B(self.comb_real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A, self.A_real_attr), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        # add reid loss to update the G_B(LR-HR)
        _, pred_label_real_A = torch.max(self.pred_real_A, 1)
        _, pred_label_fake_A = torch.max(self.pred_fake_A, 1)
        self.corrects_A += float(torch.sum(pred_label_real_A == self.A_label))
        self.corrects_B += float(torch.sum(pred_label_fake_A == self.B_label))

        # add reid loss to update the G_B(LR-HR)
        loss_reid_real_A = self.criterionReid(self.pred_real_A, self.A_label)
        loss_reid_fake_A = self.criterionReid(self.pred_fake_A, self.B_label)
        self.loss_reid = loss_reid_real_A + loss_reid_fake_A
        self.loss_G = self.loss_G + self.loss_reid

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
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
