from .base_model import BaseModel
from . import networks
from . import networks_reid
import torch


class TestReidSRModel(BaseModel):
    def name(self):
        return 'TestReidSRModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_B', 'fake_A', 'GT_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_B', 'D_reid']

        self.num_attr = opt.num_attr

        # low-resolution to high-resolution
        # self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
        #                               opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc + opt.num_attr, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_reid = networks_reid.ft_net(opt.num_classes)
        self.netD_reid = self.netD_reid.to(self.device)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.img = input['img'].to(self.device)
        self.img_label = input['img_label'].to(self.device)
        self.image_paths = input['img_paths']  # list
        # for SR real_B
        self.img_attr = input['img_attr'].to(self.device)
        self.GT_B = input['GT_img'].to(self.device)

    def SR_B(self):
        self.real_B = self.img
        self.B_real_attr = self.img_attr

        # combine the attributes
        B_real_attr = torch.unsqueeze(self.B_real_attr, 2)
        B_real_attr = B_real_attr.repeat(1, 1, self.real_B.size()[2] * self.real_B.size()[3])
        B_real_attr = torch.reshape(B_real_attr, (-1, self.num_attr, self.real_B.size()[2], self.real_B.size()[3]))
        B_real_attr = B_real_attr.float()
        comb_input_real = torch.cat([B_real_attr, self.real_B], 1)

        self.fake_A = self.netG_B(comb_input_real)
        # set the fake_A for reid
        self.img = self.fake_A

    def psnr_eval(self):
        # compute the PSNR for the test
        self.bicubic_psnr = networks.compute_psnr(self.GT_B, self.real_B)
        self.psnr = networks.compute_psnr(self.GT_B, self.fake_A)

    def ssim_eval(self):
        self.bicubic_ssim = networks.compute_ssim(self.GT_B, self.real_B)
        self.ssim = networks.compute_ssim(self.GT_B, self.fake_A)

    def extract_features(self):
        # Remove the final fc layer and classifier layer
        self.netD_reid.model.fc = torch.nn.Sequential()
        self.netD_reid.classifier = torch.nn.Sequential()
        # self.netD_reid.train(False)  # Set model to evaluate mode
        self.netD_reid = self.netD_reid.eval()

        # extract_feature
        f = self.netD_reid(self.img)  # A_label HR

        # norm feature
        fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
        f = f.div(fnorm.expand_as(f))

        # f = f.squeeze().data.cpu()
        f = f.data.cpu()
        self.features = torch.cat((self.features, f), 0)
