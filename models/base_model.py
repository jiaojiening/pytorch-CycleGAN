import os
import torch
from collections import OrderedDict
from . import networks
from . import networks_reid


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_dir_SR = os.path.join(opt.checkpoints_dir, opt.SR_name)
        self.save_dir_reid = os.path.join(opt.checkpoints_dir, opt.reid_name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        # add some attributes here
        self.optimizers = []
        self.optimizer_reid =[]
        self.psnr = 0
        self.bicubic_psnr = 0
        self.ssim = 0
        self.bicubic_ssim = 0
        self.corrects_A = 0
        self.corrects_B = 0
        self.num_attr = opt.num_attr
        self.corrects_attr_A = opt.num_attr * [0.0]
        self.corrects_attr_B = self.num_attr * [0.0]
        self.features = torch.FloatTensor()


    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def extract_features(self):
        pass

    def SR_B(self):
        pass

    def psnr_eval(self):
        pass

    def ssim_eval(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            if len(self.optimizer_reid) > 0:
                self.schedulers.append(networks_reid.get_scheduler(self.optimizer_reid[0], opt))
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def setup_attr(self, opt):
        load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
        self.load_reid_networks(load_suffix)

    # load pre-trained networks
    # initialization for different stage in joint training,
    # only for the joint training phase, not for test phase
    def setup_joint_training(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            if len(self.optimizer_reid) > 0:
                self.schedulers.append(networks_reid.get_scheduler(self.optimizer_reid[0], opt))
        load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
        if opt.stage == 0:
            # load the pre-trained SR model, jointly train the reid and SR model
            self.load_SR_networks(load_suffix)
        elif opt.stage == 1:
            # load the pre-trained reid model and fix the reid model, just train the SR model
            self.load_reid_networks(load_suffix)
        elif opt.stage ==2:
            # load the pre-trained reid model and the SR model
            self.load_reid_networks(load_suffix)
            self.load_SR_networks(load_suffix)
        else:
            assert ('unknown joint training stage')
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            # self.forward()
            self.SR_B()
            self.psnr_eval()
            self.ssim_eval()

    def test_SR(self):
        with torch.no_grad():
            self.SR_B()
            self.psnr_eval()
            self.ssim_eval()

    def test_reid(self):
        with torch.no_grad():
            self.extract_features()

    def get_features(self):
        features = self.features
        self.features = torch.FloatTensor()
        return features

    def get_psnr(self):
        return self.bicubic_psnr, self.psnr

    def get_ssim(self):
        return self.bicubic_ssim, self.ssim

    # update learning rate (called once every epoch)
    def compute_corrects(self):
        corrects_A = self.corrects_A
        corrects_B = self.corrects_B
        corrects_attr_A = self.corrects_attr_A
        corrects_attr_B = self.corrects_attr_B
        self.corrects_A = 0
        self.corrects_GT_A = 0
        self.corrects_B = 0
        self.corrects_attr_A = self.num_attr * [0.0]
        self.corrects_attr_B = self.num_attr * [0.0]
        return corrects_A, corrects_B, corrects_attr_A, corrects_attr_B

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        if len(self.optimizers) > 0:
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
        if len(self.optimizer_reid) > 0:
            reid_lr = self.optimizer_reid[0].param_groups[0]['lr']
            print('reid learning rate = %.7f' % reid_lr)

    # # update reid learning rate (called once every epoch)
    # def update_reid_learning_rate(self):
    #     # update the reid learing rate scheduler
    #     self.exp_lr_scheduler.step()
    #     reid_lr = self.optimizer_D_reid.param_groups[0]['lr']
    #     print('reid learning rate = %.7f' % reid_lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # if name == 'D_reid':
                    #     torch.save(net.cpu().state_dict(), save_path)
                    # else:
                    #     torch.save(net.module.cpu().state_dict(), save_path)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        # print(module.__class__.__name__)
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # load the G_A, D_A, G_B, D_B networks
    def load_SR_networks(self, epoch):
        for name in self.model_names:
            if name == 'D_reid':
                continue
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir_SR, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    # print(key)
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # load the reid networks
    def load_reid_networks(self, epoch):
        for name in ['D_reid']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir_reid, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    # print(key)
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                if self.opt.model == 'reid_attr':
                    # the loaded state_dict have partial params of the net, which is missing some keys
                    state = net.state_dict()
                    state.update(state_dict)
                    net.load_state_dict(state)
                else:
                    net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
