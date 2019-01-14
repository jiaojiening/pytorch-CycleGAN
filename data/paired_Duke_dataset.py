from __future__ import division
import os.path
from data.base_dataset import BaseDataset, get_transforms_reid, get_transforms_LR_reid, get_transforms_norm_reid
from data.image_folder import make_reid_dataset, find_all_index
from PIL import Image
import random
import numpy as np
import re
from scipy.io import loadmat


class PairedDukeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--up_scale', type=int, default=4, help='up_scale of the image super-resolution')
        parser.add_argument('--num_attr', type=int, default=23, help='the number of the attributes')
        parser.add_argument('--resize_h', type=int, default=256, help='the size of the height should be resized')
        parser.add_argument('--resize_w', type=int, default=128, help='the size of the width should be resized')
        parser.add_argument('--num_classes', type=int, default=702, help='The total num of the id classes ')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.dataPath = '/home/share/jiening/dgd_datasets/raw'
        self.root = opt.dataroot    # opt.dataroot = DukeMTMC-reID

        # load the attributes from the formatted attributes file, total 23 attributes
        self.attrFile = os.path.join(self.dataPath, opt.dataroot, 'Duke_attributes.mat')  # get the attributes mat file
        self.total_attr = loadmat(self.attrFile)
        self.train_attr = self.total_attr['train_attr']  # 702 * 23
        self.test_attr = self.total_attr['test_attr']    # 1110 * 23

        # split the A and B set without overlap
        if opt.phase == 'train':
            # ---------------------------------------
            # train_all (need to split A and B)
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

            self.A_attr = []
            for i in self.A_labels:
                self.A_attr.append(self.train_attr[i])
            self.B_attr = []
            for i in self.B_labels:
                self.B_attr.append(self.train_attr[i])

        else:
            # -----------------------------------------
            # query (test B) LR
            self.dir_query = os.path.join(self.dataPath, opt.dataroot, 'query')  # images in the query
            self.query_paths, self.query_labels = make_reid_dataset(self.dir_query)
            self.query_num = len(self.query_paths)  # 2228
            print('total %d images in query' % self.query_num)

            # -----------------------------------------
            # gallery (test A) HR
            self.dir_gallery = os.path.join(self.dataPath, opt.dataroot, 'bounding_box_test')
            self.gallery_paths, self.gallery_labels = make_reid_dataset(self.dir_gallery)
            self.gallery_num = len(self.gallery_paths)  # 17661
            print('total %d images in bounding_box_test' % self.gallery_num)

            self.test_attr_map = {}
            # the query_labels are included in the gallery_labels
            for i, label in enumerate(list(np.unique(np.array(self.gallery_labels)))):
                self.test_attr_map[label] = i

            self.A_paths = self.gallery_paths
            self.B_paths = self.query_paths
            self.A_labels = self.gallery_labels
            self.B_labels = self.query_labels

            self.A_attr = []
            for i in self.gallery_labels:
                # obtain the according id
                attr_id = self.test_attr_map[i]
                self.A_attr.append(self.test_attr[attr_id])
            self.B_attr = []
            for i in self.query_labels:
                # obtain the according id
                attr_id = self.test_attr_map[i]
                self.B_attr.append(self.test_attr[attr_id])

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(self.A_size)
        print(self.B_size)

        # A: high_resolution, B: low_resolution
        # opt.fineSize = 128, opt.loadSize = 158, need to modify
        self.transform = get_transforms_reid(opt)
        self.transform_LR = get_transforms_LR_reid(opt)
        self.transform_norm = get_transforms_norm_reid()

    def __getitem__(self, index):
        # we want to learn BtoA, e.g., low-resolution to high-resolution
        B_path = self.B_paths[index % self.B_size]
        B_label = self.B_labels[index % self.B_size]

        # find all the same label as B_label in the A_labels
        all_index_A = find_all_index(self.A_labels, B_label)

        # choose the image in A(HR) for the first index or randomly
        # default: randomly, serial_batches = False
        if self.opt.serial_batches:
            index_A = all_index_A[0]
        else:
            index_A = all_index_A[random.randint(0, len(all_index_A) - 1)]
        A_path = self.A_paths[index_A]
        A_label = self.A_labels[index_A]

        assert A_label == B_label, 'A_label must be equal to B_label'

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        GT_A = self.transform_LR(A)  # ground-truth low-resolution
        # normalize the images
        A = self.transform_norm(A)
        GT_A = self.transform_norm(GT_A)

        GT_B = self.transform(B_img)  # ground-truth high-resolution
        B = self.transform_LR(GT_B)  # produce the low-resolution images of the GT_B
        # normalize the images
        GT_B = self.transform_norm(GT_B)
        B = self.transform_norm(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        # need to add the attributes of the high_resolution(A)
        A_real_attr = self.A_attr[index_A][:]
        # sample a fake attributes for A
        fake_index = np.random.permutation(self.A_size)[0]
        A_fake_attr = self.A_attr[fake_index][:]
        while list(A_fake_attr) == list(A_real_attr):
            fake_index = np.random.permutation(self.A_size)[0]
            A_fake_attr = self.A_attr[fake_index][:]

        # only have the attributes of low-resolution(B) images
        B_real_attr = self.B_attr[index % self.B_size]
        # sample a fake attributes for B
        fake_index = np.random.permutation(self.B_size)[0]
        B_fake_attr = self.B_attr[fake_index][:]
        while list(B_fake_attr) == list(B_real_attr):
            fake_index = np.random.permutation(self.B_size)[0]
            B_fake_attr = self.B_attr[fake_index][:]

        assert (A_real_attr == B_real_attr).all(), 'A_real_attr must be equal to B_real_attr'

        # # change the attributes value, (1,2)->(-1,1)
        # A_real_attr = list((np.array(A_real_attr) - 1.5) * 0.5)
        # A_fake_attr = list((np.array(A_fake_attr) - 1.5) * 0.5)
        # B_real_attr = list((np.array(B_real_attr) - 1.5) * 0.5)

        return {'A': A, 'B': B,
                'GT_A': GT_A, 'GT_B': GT_B,
                'A_paths': A_path, 'B_paths': B_path,
                'A_real_attr': A_real_attr, 'A_fake_attr': A_fake_attr,
                'B_real_attr': B_real_attr, 'B_fake_attr': B_fake_attr,
                'A_label': A_label, 'B_label': B_label}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'PairedDukeDataset'
