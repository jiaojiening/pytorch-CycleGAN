from __future__ import division
import os.path
# from data.base_dataset import BaseDataset, get_transform
from data.base_dataset import BaseDataset, get_transforms, get_transform_LR, get_transform_norm
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import re


class CelebADataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--up_scale', type=int, default=4, help='up_scale of the image super-resolution')
        parser.add_argument('--num_attr', type=int, default=18, help='the number of the attributes')
        # the same with the opt.fineSize
        # parser.add_argument('--resize_h', type=int, default=128, help='the size of the height should be resized')
        # parser.add_argument('--resize_w', type=int, default=128, help='the size of the width should be resized')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    # opt.dataroot = CelebA
        self.dir_total = os.path.join(opt.dataroot, 'img_align_celeba')   # total images in this path
        self.total_paths = make_dataset(self.dir_total)
        self.total_num = len(self.total_paths)   # 202599
        print('total %d images in img_align_celeba'% self.total_num)

        self.attrFile = os.path.join(opt.dataroot, 'list_attr_celeba.txt')   # get the attributes file
        # Index of the attributes from celebA that will be ignored
        self.attrFil = [1,2,3,4,7,8,11,14,15,17,20,24,25,26,28,30,31,35,37,38,39,40]
        # get the selected attributes according to the total_paths
        self.imLabels = self.get_attributes()

        # split the train and test set
        # may need to split according to the person id
        self.testSetSize = 19961  # follow the IcGANs
        self.randIdx_file = os.path.join(opt.dataroot, 'randIdx.npy')
        if os.path.exists(self.randIdx_file):
            randIdx = np.load(self.randIdx_file)
        else:
            randIdx = np.random.permutation(self.total_num)
            np.save(self.randIdx_file,randIdx)
        trainIdx = randIdx[:self.total_num-self.testSetSize]
        trainIdx = list(trainIdx)
        # can be both low-resolution
        testIdx = randIdx[self.total_num-self.testSetSize:]
        testIdx = list(testIdx)

        # split the A and B set without overlap
        if opt.phase == 'train':
            A_Idx = trainIdx[:len(trainIdx)//2]
            B_Idx = trainIdx[len(trainIdx)//2:]
        else:
            A_Idx = testIdx[:len(testIdx) // 2]
            B_Idx = testIdx[len(testIdx) // 2:]

        self.A_paths = [self.total_paths[i] for i in A_Idx]
        self.B_paths = [self.total_paths[i] for i in B_Idx]
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(self.A_size)
        print(self.B_size)

        # A: high_resolution, B: low_resolution
        # opt.fineSize = 128, opt.loadSize = 158, need to modify
        self.transform_A = get_transforms(opt, type='A')
        self.transform_B = get_transforms(opt, type='B')
        self.transform_LR = get_transform_LR(opt)
        self.transform_norm = get_transform_norm()

        # add the A_attr (attributes of the high-resolution images)
        self.A_attr = [self.imLabels[i][:] for i in A_Idx]
        self.B_attr = [self.imLabels[i][:] for i in B_Idx]


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        GT_B = self.transform_B(B_img)  # ground-truth high-resolution

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
        A_real_attr = self.A_attr[index % self.A_size][:]
        # sample a fake attributes for A
        fake_index = np.random.permutation(self.A_size)[0]
        A_fake_attr = self.A_attr[fake_index][:]
        while list(A_fake_attr) == list(A_real_attr):
            fake_index = np.random.permutation(self.A_size)[0]
            A_fake_attr = self.A_attr[fake_index][:]

        B_real_attr = self.B_attr[index_B]

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path,
                'GT_B': GT_B,
                'A_real_attr': A_real_attr, 'A_fake_attr': A_fake_attr,
                'B_real_attr': B_real_attr}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'CelebADataset'

    # self.imLabels = get_attributes(self.total_paths, self.total_num)
    def get_attributes(self):
        f = open(self.attrFile, 'r')
        f.readline()  # skip the 1st line
        rawLabelHeader = f.readline() # 2nd line is the header
        rawLabelHeader = rawLabelHeader.split(' ')
        # select 18 attributes from the total 40 attributes
        LabelHeader = [rawLabelHeader[i-1] for i in range(1,41) if i not in self.attrFil]
        print(len(LabelHeader))

        imLabels = []
        count = 0
        for line in f.readlines():
            # unify the space with the character @
            line = re.sub(' +', '@', line)
            line = line.split('@')
            file_name = line[0]
            assert file_name == os.path.basename(self.total_paths[count])
            # select 18 attributes value from the total 40 attributes
            attr = [int(line[i]) for i in range(1,41) if i not in self.attrFil]
            imLabels.append(attr)
            count += 1
        imLabels = np.array(imLabels)
        print(np.shape(imLabels))
        return imLabels


