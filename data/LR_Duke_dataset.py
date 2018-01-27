from __future__ import division
import os.path
from data.base_dataset import BaseDataset, get_transforms_reid, get_transforms_LR_reid, get_transforms_norm_reid
from data.image_folder import make_reid_dataset
from PIL import Image
import random
import numpy as np
import re
from scipy.io import loadmat


class LRDukeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        Duke_attr_class_num = [2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # Duke_attr_mask = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        Duke_attr_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        parser.add_argument('--up_scale', type=int, default=4, help='up_scale of the image super-resolution')
        parser.add_argument('--num_attr', type=int, default=23, help='the number of the attributes')
        parser.add_argument('--resize_h', type=int, default=256, help='the size of the height should be resized')
        parser.add_argument('--resize_w', type=int, default=128, help='the size of the width should be resized')
        parser.add_argument('--num_classes', type=int, default=702, help='the total num of the id classes')
        parser.add_argument('--attr_class_num', nargs='+', type=int, help='the number of classes of each attributes')
        parser.set_defaults(attr_class_num=Duke_attr_class_num)
        parser.add_argument('--attr_mask', nargs='+', type=int, help='ignore some attributes')
        parser.set_defaults(attr_mask=Duke_attr_mask)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.dataPath = '/home/share/jiening/dgd_datasets/raw'
        self.root = opt.dataroot    # opt.dataroot = DukeMTMC-reID

        # load the attributes from the formatted attributes file, total 23 attributes
        # the number of classes of each attributes
        # duke_attribute.train.top: 0, 1, 2 (index = 7)  id:[370:165, 679:326], attr:[1, 2]
        # self.attr_class_num = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.attrFile = os.path.join(self.dataPath, opt.dataroot, 'Duke_attributes.mat')  # get the attributes mat file
        self.total_attr = loadmat(self.attrFile)
        self.train_attr = self.total_attr['train_attr']  # 702 * 23
        self.test_attr = self.total_attr['test_attr']    # 1110 * 23

        if opt.phase == 'train':
            # ---------------------------------------
            # train_all
            self.dir_train = os.path.join(self.dataPath, opt.dataroot, 'bounding_box_train')
            self.train_paths, self.train_labels = make_reid_dataset(self.dir_train)
            self.train_num = len(self.train_paths)  # 16522
            print('total %d images in bounding_box_train' % self.train_num)

            self.train_id_map = {}
            for i, label in enumerate(list(np.unique(np.array(self.train_labels)))):
                self.train_id_map[label] = i
            # map the train_labels to train_id_labels start from zeros (0-702)
            train_id_labels = list(map(lambda x: self.train_id_map[x], self.train_labels))

            # random half split the A and B in train
            self.randIdx_file = os.path.join(self.dataPath, opt.dataroot, 'randIdx_Duke.npy')
            if os.path.exists(self.randIdx_file):
                randIdx = np.load(self.randIdx_file)
            else:
                randIdx = np.random.permutation(self.train_num)
                np.save(self.randIdx_file, randIdx)
            print(len(randIdx))
            A_Idx = randIdx[:len(randIdx) // 2]
            B_Idx = randIdx[len(randIdx) // 2:]

            self.A_paths = [self.train_paths[i] for i in A_Idx]
            self.B_paths = [self.train_paths[i] for i in B_Idx]
            self.A_labels = [train_id_labels[i] for i in A_Idx]
            self.B_labels = [train_id_labels[i] for i in B_Idx]

            # check that both the HR and LR images of each id
            print(len(set(self.A_labels)))  # 702
            print(len(set(self.B_labels)))  # 702

            # self.A_attr = []
            # for i in self.A_labels:
            #     self.A_attr.append(self.train_attr[i])
            # self.B_attr = []
            # for i in self.B_labels:
            #     self.B_attr.append(self.train_attr[i])

            self.img_paths = self.train_paths
            self.img_labels = train_id_labels
            self.img_size = len(self.train_paths)

        # A: high_resolution, B: low_resolution
        # opt.fineSize = 128, opt.loadSize = 158, need to modify
        self.transform = get_transforms_reid(opt)
        self.transform_LR = get_transforms_LR_reid(opt)
        self.transform_norm = get_transforms_norm_reid()


    def __getitem__(self, index):
        # choose the image in order or randomly
        # default: randomly, serial_batches = False
        if self.opt.serial_batches:
            index = index % self.img_size
        else:
            index = random.randint(0, self.img_size - 1)
        img_label = self.img_labels[index]
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        if img_path in self.B_paths:
            img = self.transform_LR(img)
        img = self.transform_norm(img)

        # print(img_label)
        img_attr = self.train_attr[img_label]

        return {'img': img, 'img_paths': img_path,
                'img_attr': img_attr,
                'img_label': img_label}

    def __len__(self):
        return self.img_size

    def name(self):
        return 'LRDukeDataset'
