import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_reid


class SRReidModel(BaseModel):
    def name(self):
        return 'SRReidModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # reid parameters, put the parameter num_classes in the dataset
        parser.add_argument('--droprate', type=float, default=0.5, help='the dropout ratio in reid model')
        parser.add_argument('--NR', action='store_true', help='use the normal resolution dataset')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['reid', 'reid_real_A', 'reid_real_B']
        self.loss_names = ['reid']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            visual_names_A = ['real_A']
            visual_names_B = ['real_B', 'LR_B', 'SR_B']

            self.visual_names = visual_names_A + visual_names_B
        else:
            self.visual_names=['img']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_reid']

        # define the re-id network
        # Load a pre-trained resnet model and reset the final connected layer
        # the dropout layer is in the classifier
        if self.isTrain:
            self.netD_reid = networks_reid.ft_net(opt.num_classes, opt.droprate)
        else:
            self.netD_reid = networks_reid.ft_net(opt.num_classes)

        # use gpu
        self.netD_reid = self.netD_reid.to(self.device)
        # self.netD_reid = torch.nn.DataParallel(self.netD_reid, opt.gpu_ids)

        if self.isTrain:
            self.criterionReid = torch.nn.CrossEntropyLoss()
            # initialize reid optimizer
            ignored_params = list(map(id, self.netD_reid.model.fc.parameters())) + \
                             list(map(id, self.netD_reid.classifier.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.netD_reid.parameters())
            self.optimizer_D_reid = torch.optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.reid_lr},
                {'params': self.netD_reid.classifier.parameters(), 'lr': opt.reid_lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

            self.optimizer_reid.append(self.optimizer_D_reid)

    def set_input(self, input):
        if self.isTrain:
            self.real_A = input['A'].to(self.device)   # high-resolution
            # self.GT_A = input['GT_A'].to(self.device)  # low-resolution

            # train on the normal resolution B set if NR
            # super-resolution B image, e.g.,fake_A
            self.real_B = input['B'].to(self.device)  # low-resolution

            self.SR_B = input['SR_B'].to(self.device)
            self.LR_B = input['LR_B'].to(self.device)

            # get the id label for person reid
            self.A_label = input['A_label'].to(self.device)
            self.B_label = input['B_label'].to(self.device)

            # self.A_paths = input['A_paths']
            # self.B_paths = input['B_paths']

        else:
            if self.opt.dataset_type == 'B'and self.opt.NR:
                # if NR, dataset B use the high-resolution GT_img
                self.img = input['GT_img'].to(self.device)
            else:
                self.img = input['img'].to(self.device)
            # print(self.img.size())
            self.img_label = input['img_label'].to(self.device)
            self.image_paths = input['img_paths']  # list

    def forward(self):
        if self.isTrain:
            # self.netD_reid.train(True)   # Set model to training mode
            self.netD_reid = self.netD_reid.train()

            # baseline
            # self.imags = torch.cat([self.real_A, self.LR_B], 0)
            # self.imags = torch.cat([self.real_A, self.real_B], 0)
            self.imags = torch.cat([self.real_A, self.SR_B], 0)
            self.labels = torch.cat([self.A_label, self.B_label], 0)
            self.pred_imgs = self.netD_reid(self.imags)

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

    def backward_G(self):
        _, pred_label_imgs = torch.max(self.pred_imgs, 1)
        self.corrects += float(torch.sum(pred_label_imgs == self.labels))
        self.loss_reid = self.criterionReid(self.pred_imgs, self.labels)
        # print(self.loss_reid)

        self.loss_G = self.loss_reid
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_D_reid.zero_grad()
        self.backward_G()
        self.optimizer_D_reid.step()
