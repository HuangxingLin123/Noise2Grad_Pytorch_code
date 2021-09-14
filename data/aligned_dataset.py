import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import cv2




class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.istrain = opt.isTrain
        self.root = opt.dataroot
        self.dir_X = os.path.join(opt.dataroot)
        self.X_paths = sorted(make_dataset(self.dir_X))



    def __getitem__(self, index):
        if self.istrain:
            X_path = self.X_paths[index]
            X = cv2.imread(X_path)
            (h, w, n) = X.shape

            width = 128

            h_off = random.randint(0, h - width)
            w_off = random.randint(0, w - width)

            X = X[h_off:h_off + width, w_off:w_off + width]

            X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

            rr = random.randint(0, 3)

            if rr == 1:
                X = cv2.flip(X, 0)
            elif rr == 2:
                X = cv2.flip(X, 1)
            elif rr == 3:
                X = cv2.flip(X, -1)
            else:
                pass

            ind = random.randint(0, 5000)
            Y = cv2.imread('./datasets/train/reference_clean_image/' + str(ind) + '.png')

            (h3, w3, _) = Y.shape
            h3_off = random.randint(0, h3 - width)
            w3_off = random.randint(0, w3 - width)

            Y = Y[h3_off:h3_off + width, w3_off:w3_off + width]
            Y = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)

            X = transforms.ToTensor()(X)

            Y = transforms.ToTensor()(Y)

            return {'X': X, 'Y': Y, 'X_paths': X_path}
        else:

            X_path = self.X_paths[index]
            X = cv2.imread(X_path)
            X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

            X = transforms.ToTensor()(X)

            return {'X': X, 'X_paths': X_path}



    def __len__(self):
        return len(self.X_paths)

    def name(self):
        return 'AlignedDataset'
