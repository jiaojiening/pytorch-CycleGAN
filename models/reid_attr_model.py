import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import networks_reid


class ReidAttrModel(BaseModel):
    def name(self):
        return 'ReidAttrModel'

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
            parser.add_argument('--lambda_reid', type=float, default=1.0,
                                help='the weight of the reid loss.')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['reid', 'attr']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A']
        visual_names_B = ['real_B']

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_reid']
        # self.model_names = ['Reid']

        # self.num_attr = opt.num_attr

        # define the re-id network
        # Load a pretrained resnet model and reset the final connected layer
        # the dropout layer is in the classifier
        if self.isTrain:
            self.netD_reid = networks_reid.ft_attr_net(opt.num_classes, opt.attr_class_num, opt.droprate)
        else:
            self.netD_reid = networks_reid.ft_attr_net(opt.num_classes, opt.attr_class_num)

        # use gpu
        self.netD_reid = self.netD_reid.to(self.device)
        # self.netD_reid = torch.nn.DataParallel(self.netD_reid, opt.gpu_ids)

        if self.isTrain:
            self.criterionReid = torch.nn.CrossEntropyLoss()
            # initialize reid optimizer
            ignored_params = list(map(id, self.netD_reid.model.fc.parameters())) + \
                             list(map(id, self.netD_reid.classifier.parameters()))

            # # print the parameter names
            # params = self.netD_reid.classifier.state_dict()
            # for k, v in params.items():
            #     print(k)

            base_params = filter(lambda p: id(p) not in ignored_params, self.netD_reid.parameters())
            opt.reid_lr = opt.reid_lr * 0.1
            self.optimizer_D_reid = torch.optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.reid_lr},
                # {'params': self.netD_reid.classifier.parameters(), 'lr': opt.reid_lr}
                {'params': self.netD_reid.classifier.classifier.parameters(), 'lr': 0.1 * opt.reid_lr},
                {'params': self.netD_reid.classifier.attr_classifiers.parameters(), 'lr': opt.reid_lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

            self.optimizer_reid.append(self.optimizer_D_reid)
            # Decay learning rate by a factor of 0.1 every 40 epochs
            # self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D_reid,
            #                                                         step_size=40, gamma=0.1)
            # self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D_reid,
                                                               # step_size=20, gamma=0.1)

            # load the pre-trained reid model
            self.setup_attr(opt)

    def set_input(self, input):
        if self.isTrain:
            # AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A'].to(self.device)
            # train on the normal resolution B set
            self.real_B = input['B' if not self.opt.NR else 'GT_B'].to(self.device)

            # get the id label for person reid
            self.A_label = input['A_label'].to(self.device)
            self.B_label = input['B_label'].to(self.device)
            self.A_real_attr = input['A_real_attr'].to(self.device)
            self.B_real_attr = input['B_real_attr'].to(self.device)
        else:
            self.img = input['img' if not self.opt.NR else 'GT_img'].to(self.device)
            self.img_label = input['img_label'].to(self.device)
            self.image_paths = input['img_paths']  # list

    def forward(self):
        # self.netD_reid.train(True)   # Set model to training mode
        self.netD_reid = self.netD_reid.train()

        # training: 1 * num_classes prediction vector,
        # test: 1 * 2048 feature vector
        self.pred_real_A, self.attr_pred_real_A = self.netD_reid(self.real_A)  # A_label HR
        self.pred_real_B, self.attr_pred_real_B = self.netD_reid(self.real_B)  # B_label LR

    def extract_features(self):
        # Remove the final fc layer and classifier layer
        # self.netD_reid.model.fc = torch.nn.Sequential()
        # self.netD_reid.classifier = torch.nn.Sequential()
        # self.netD_reid.train(False)  # Set model to evaluate mode
        self.netD_reid = self.netD_reid.eval()

        # extract_feature
        # x1, x2 = self.netD_reid(self.img)  # A_label HR
        self.netD_reid(self.img)
        f = self.netD_reid.get_feature()

        # norm feature
        fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
        f = f.div(fnorm.expand_as(f))

        # f = f.squeeze().data.cpu()
        f = f.data.cpu()
        self.features = torch.cat((self.features, f), 0)

    def backward_loss(self):
        lambda_reid = self.opt.lambda_reid

        _, pred_label_real_A = torch.max(self.pred_real_A, 1)
        _, pred_label_real_B = torch.max(self.pred_real_B, 1)
        self.corrects_A += float(torch.sum(pred_label_real_A == self.A_label))
        self.corrects_B += float(torch.sum(pred_label_real_B == self.B_label))

        # add reid loss to update the G_B(LR-HR)
        self.loss_reid_real_A = self.criterionReid(self.pred_real_A, self.A_label)
        self.loss_reid_real_B = self.criterionReid(self.pred_real_B, self.B_label)
        # self.loss_reid = self.loss_reid_real_A + self.loss_reid_real_B
        self.loss_reid = (self.loss_reid_real_A + self.loss_reid_real_B)/2.0

        # add the attributes loss
        loss_attr_A = 0
        # print(self.A_real_attr.size())    #[batch_szie, 23]
        for index, attr_pred in enumerate(self.attr_pred_real_A):
            # print(attr_pred.size())       # [[batch_size, 2]]
            # real_attr = (self.A_real_attr[:, index] - 1).long()
            real_attr = (self.A_real_attr[:, index] - torch.min(self.A_real_attr[:, index])).long()
            # loss_attr_A +=  self.criterionReid(attr_pred, real_attr)
            loss_attr_A += self.criterionReid(attr_pred, real_attr) * self.opt.attr_mask[index]
            _, pred_attr = torch.max(attr_pred, 1)
            self.corrects_attr_A[index] += float(torch.sum(pred_attr == real_attr))
        # loss_attr_A = loss_attr_A / len(self.attr_pred_real_A)
        loss_attr_A = loss_attr_A / sum(self.opt.attr_mask)

        loss_attr_B = 0
        for index, attr_pred in enumerate(self.attr_pred_real_B):
            # real_attr = (self.B_real_attr[:, index] - 1).long()
            real_attr = (self.B_real_attr[:, index] - torch.min(self.B_real_attr[:, index])).long()
            # loss_attr_B += self.criterionReid(attr_pred, real_attr)
            loss_attr_B += self.criterionReid(attr_pred, real_attr) * self.opt.attr_mask[index]
            _, pred_attr = torch.max(attr_pred, 1)
            self.corrects_attr_B[index] += float(torch.sum(pred_attr == real_attr))
        # loss_attr_B = loss_attr_B / len(self.attr_pred_real_B)
        loss_attr_B = loss_attr_B / sum(self.opt.attr_mask)

        # self.loss_attr = loss_attr_A + loss_attr_B
        self.loss_attr = (loss_attr_A + loss_attr_B)/2.0
        self.loss = lambda_reid*self.loss_reid + self.loss_attr

        self.loss.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_D_reid.zero_grad()
        self.backward_loss()
        self.optimizer_D_reid.step()
