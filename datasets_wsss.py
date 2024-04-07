import torch
import os
import torchvision.transforms as transforms
from dataset.transforms import *

from skimage import io
from torch.utils.data import Dataset

import PIL
import numpy as np
# import pandas as pd

class Dataset_train(Dataset):
    def __init__(self, dataset_size, device, dataset, test_on_train, stage):
        super(Dataset_train, self).__init__()
        
        self.root = f'data/{dataset}/training'
        self.data = sorted(os.listdir(self.root)) # name
        
        if dataset == 'LUAD-HistoSeg':
            self.label = [file[:-4].split(']')[0].split('[')[-1].split(' ') for file in self.data]
        elif dataset == 'BCSS-WSSS':
            self.label = [[x for x in file[:-4].split(']')[0].split('[')[-1]] for file in self.data]
        else:
            raise NotImplementedError
        
        self.device = device
        self.size = dataset_size
        self.stage = stage
        self.dataset = dataset
        
        if test_on_train:
            self.transforms = Compose([
                ToTensor(),
                Normalize()
            ])
        else:
            self.transforms = Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomGaussianBlur(),
                ToTensor(),
                Normalize(),
            ])
        
        self.strong_transforms = Compose([
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.3),
            RandomGaussianBlur(),
            ToTensor(),
            RandomErasing(),
            Normalize()
        ])

    def __getitem__(self, index):
        image, img_show = self.read(self.root, self.data[index])
        strong_image = self.readStrong(self.root, self.data[index])
        # if self.stage == 0:
        #     label = torch.Tensor([int(self.label[index][0]),int(self.label[index][1]),int(self.label[index][2]),int(self.label[index][3])])
        # elif self.stage == 1:
        #     name = self.data[index].split('.')[0]
        #     label = np.load(f'./wsss_results/{self.dataset}/{name}.npy')

        label = torch.Tensor([int(self.label[index][0]),int(self.label[index][1]),int(self.label[index][2]),int(self.label[index][3])])
        # label = label[0:2]
        # label[0] = 0
        # label[1] = 0

        return image, strong_image, label, {"img_show": img_show, "file_name": self.data[index]}

    def __len__(self):
        return len(self.data)

    def read(self, path, name):
        img = PIL.Image.open(os.path.join(path, name))
        img, img_show = self.transforms(img, img.copy())
        return img, img_show
    
    def readStrong(self, path, name):
        img = PIL.Image.open(os.path.join(path, name))
        img = self.strong_transforms(img)
        return img

class Dataset_valid(Dataset):
    def __init__(self, dataset_size, device, dataset):
        super(Dataset_valid, self).__init__()
        self.root = f'data/{dataset}/val/'
        self.data = sorted(os.listdir(self.root + 'img')) # name   
        
        self.device = device
        self.size = dataset_size
        
        self.transforms_test = Compose([
            ToTensor(),
            Normalize()
        ])
        self.transforms_grdth = Compose([
            ToTensor()
        ])

    def __getitem__(self, index):
        
        image = self.read(self.root, 'img/' + self.data[index], 'test')
        grdth = self.read(self.root, 'mask/' + self.data[index], 'grdth') * 255

        return image, grdth

    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name))

        if norm == 'test':
            img = self.transforms_test(img)

        elif norm == 'grdth':
            img = self.transforms_grdth(img)

        return img


class Dataset_test(Dataset):
    def __init__(self, dataset_size, device, dataset):
        super(Dataset_test, self).__init__()
        self.root = f'data/{dataset}/test/'
        self.data = sorted(os.listdir(self.root + 'img')) # name  
        
        self.device = device
        self.size = dataset_size
        
        self.transforms_test = Compose([
            ToTensor(),
            Normalize()
        ])
        self.transforms_grdth = Compose([
            ToTensor()
        ])

    def __getitem__(self, index):

        image = self.read(self.root, 'img/' + self.data[index], 'test')
        label = self.read(self.root, 'mask/' + self.data[index], 'grdth') * 255
        image_show = self.read(self.root, 'img/' + self.data[index])

        return image, label, image_show

    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name))

        if norm == 'test':
            img = self.transforms_test(img)

        elif norm == 'grdth':
            img = self.transforms_grdth(img)
        else:
            img = torch.from_numpy(np.array(img)).float()

        return img
    

