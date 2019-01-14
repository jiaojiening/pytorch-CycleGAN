import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_reid


class ReidModel(BaseModel):
    def name(self):
        return 'ReidModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        # reid parameters, put the parameter num_classes in the dataset
        parser.add_argument('--droprate', type=float, default=0.5, help='the dropout ratio in reid model')
        parser.add_argument('--NR', action='store_true', help='use the normal resolution dataset')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['reid']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A']
        visual_names_B = ['real_B']

        self.visual_names = visual_names_A + visual_names_B
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
                {'params': self.netD_reid.model.fc.parameters(), 'lr': opt.reid_lr},
                {'params': self.netD_reid.classifier.parameters(), 'lr': opt.reid_lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

            self.optimizer_reid.append(self.optimizer_D_reid)
            # Decay learning rate by a factor of 0.1 every 40 epochs
            # self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D_reid,
            #                                                         step_size=40, gamma=0.1)
            # self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D_reid,
                                                                    # step_size=20, gamma=0.1)

    def set_input(self, input):
        if self.isTrain:
            # AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A'].to(self.device)
            self.GT_A = input['GT_A'].to(self.device)

            # train on the normal resolution B set if NR
            self.real_B = input['B' if not self.opt.NR else 'GT_B'].to(self.device)

            # get the id label for person reid
            self.A_label = input['A_label'].to(self.device)
            self.B_label = input['B_label'].to(self.device)
        else:
            self.img = input['img' if not self.opt.NR else 'GT_img'].to(self.device)
            self.img_label = input['img_label'].to(self.device)
            self.image_paths = input['img_paths']  # list

    def forward(self):
        if self.isTrain:
            # self.netD_reid.train(True)   # Set model to training mode
            self.netD_reid = self.netD_reid.train()

            # training: 1 * num_classes prediction vector,
            # test: 1 * 2048 feature vector
            self.pred_real_A = self.netD_reid(self.real_A)  # A_label HR
            self.pred_GT_A = self.netD_reid(self.GT_A)  # A_label LR
            self.pred_real_B = self.netD_reid(self.real_B)  # B_label LR
        else:
            # Remove the final fc layer and classifier layer
            self.netD_reid.model.fc = torch.nn.Sequential()
            self.netD_reid.classifier = torch.nn.Sequential()
            # self.netD_reid.train(False)  # Set model to evaluate mode
            self.netD_reid = self.netD_reid.eval()

            # extract_feature
            f = self.netD_reid(self.img)  # A_label HR
            self.features = torch.cat((self.features, f), 0)

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
        # print(self.A_label)
        # print(self.B_label)
        _, pred_label_real_A = torch.max(self.pred_real_A, 1)
        _, pred_label_real_B = torch.max(self.pred_real_B, 1)
        # print(pred_label_real_A)
        # print(pred_label_real_B)
        self.corrects_A += float(torch.sum(pred_label_real_A == self.A_label))
        self.corrects_B += float(torch.sum(pred_label_real_B == self.B_label))

        # add reid loss to update the G_B(LR-HR)
        self.loss_reid_real_A = self.criterionReid(self.pred_real_A, self.A_label)
        self.loss_reid_real_B = self.criterionReid(self.pred_real_B, self.B_label)

        # self.loss_reid = (self.loss_reid_real_A + self.loss_reid_real_B)/2.0

        _, pred_label_GT_A = torch.max(self.pred_GT_A, 1)
        self.corrects_GT_A += float(torch.sum(pred_label_GT_A == self.A_label))
        self.loss_reid_GT_A = self.criterionReid(self.pred_GT_A, self.A_label)
        self.loss_reid = (self.loss_reid_real_A + self.loss_reid_real_B + self.corrects_GT_A) / 3.0

        self.loss_G = self.loss_reid
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_D_reid.zero_grad()
        self.backward_G()
        self.optimizer_D_reid.step()
