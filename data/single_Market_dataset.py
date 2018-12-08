import os.path
# from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transforms_reid, get_transform_LR_reid, get_transform_norm_reid
from data.image_folder import make_dataset, make_reid_dataset
from PIL import Image
from scipy.io import loadmat
import numpy as np


class SingleMarketDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--dataset_type', type=str, default='A', help='the A set')
        parser.add_argument('--up_scale', type=int, default=4, help='up_scale of the image super-resolution')
        parser.add_argument('--num_attr', type=int, default=23, help='the number of the attributes')
        parser.add_argument('--resize_h', type=int, default=256, help='the size of the height should be resized')
        parser.add_argument('--resize_w', type=int, default=128, help='the size of the width should be resized')
        parser.add_argument('--num_classes', type=int, default=751, help='The total num of the id classes ')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.dataPath = '/home/share/jiening/dgd_datasets/raw'
        # self.root = opt.dataroot    # opt.dataroot = Market-1501-v15.09.15
        if opt.dataroot == 'Market':
            self.root = 'Market-1501-v15.09.15'
        self.dataset_type = opt.dataset_type

        # load the attributes from the formatted attributes file, total 27 attributes
        self.attrFile = os.path.join(self.dataPath, self.root, 'Market_attributes.mat')  # get the attributes mat file
        self.total_attr = loadmat(self.attrFile)
        self.train_attr = self.total_attr['train_attr']  # 751 * 27
        self.test_attr = self.total_attr['test_attr']  # 750 * 27

        # load the attributes index from the index file, total 27 attributes
        self.attrIndexFile = os.path.join(self.dataPath, self.root, 'Market_index.mat')
        self.total_attrIndex = loadmat(self.attrIndexFile)
        self.train_attrIndex = self.total_attrIndex['train_index'][0]  # 751
        self.test_attrIndex = self.total_attrIndex['test_index'][0]  # 750

        # -----------------------------------------
        # query (test B) LR
        dir_query = os.path.join(self.dataPath, self.root, 'query')  # images in the query
        query_paths, query_labels = make_reid_dataset(dir_query)
        query_num = len(query_paths)  # 2228
        print('total %d images in query' % query_num)

        # -----------------------------------------
        # gallery (test A) HR
        dir_gallery = os.path.join(self.dataPath, self.root, 'bounding_box_test')
        gallery_paths, gallery_labels = make_reid_dataset(dir_gallery)
        gallery_num = len(gallery_paths)  # 17661
        print('total %d images in bounding_box_test' % gallery_num)

        self.test_attr_map = {}
        # the query_labels are included in the gallery_labels
        for i, label in enumerate(self.test_attrIndex):
            self.test_attr_map[label] = i

        if self.dataset_type == 'A':
            self.img_paths = gallery_paths
            self.img_labels = gallery_labels
            # self.img_attrs = []
            # for i in gallery_labels:
            #     # obtain the according id
            #     attr_id = self.test_attr_map[i]
            #     self.img_attrs.append(self.test_attr[attr_id])
        else:
            self.img_paths = query_paths
            self.img_labels = query_labels
            self.img_attrs = []
            for i in query_labels:
                # obtain the according id
                attr_id = self.test_attr_map[i]
                self.img_attrs.append(self.test_attr[attr_id])

        # A: high-resolution, B: low-resolution
        # TODO: in base_dataset, maybe return the transform_list
        self.transform_A = get_transforms_reid(opt, type='A')
        self.transform_B = get_transforms_reid(opt, type='B')
        self.transform_LR = get_transform_LR_reid(opt)
        self.transform_norm = get_transform_norm_reid()

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        # img = self.transform_A(img)

        if self.dataset_type == 'A':
            img = self.transform_A(img)
            img_attr = []
        else:
            # low-resolution image
            img = self.transform_B(img)
            img = self.transform_LR(img)
            img = self.transform_norm(img)
            img_attr = self.img_attrs[index]

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = tmp.unsqueeze(0)

        img_label = self.img_labels[index]
        return {'img': img, 'img_paths': img_path,
                'img_attr': img_attr,
                'img_label': img_label}

    def __len__(self):
        return len(self.img_paths)

    def name(self):
        return 'SingleMarketDataset'
