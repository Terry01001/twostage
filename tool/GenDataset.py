# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tool import custom_transforms as tr

class Stage1_InferDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.object = self.path_label()

    def __getitem__(self, index):
        fn = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_list = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                path_list.append(image_path)
        return path_list

class Stage1_TrainDataset(Dataset):
    def __init__(self, data_path, dataset=None):
        self.data_path = os.path.join(data_path, 'LUAD-HistoSeg/training/')
        self.dataset = dataset
        self.object = self.path_label()
        # print(self.data_path)
        

    def __getitem__(self, index):
        fn, label = self.object[index]
        img = Image.open(fn).convert('RGB')
        img = self.transform_tr(img)
        return fn.split('/')[-1][:-4], img, label
        
    def __len__(self):
        return len(self.object)
        
    def path_label(self):
        path_label = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                ##  Extract the image-level label from the filename
                ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
                ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
                label_str = fname.split(']')[0].split('[')[-1]
                if self.dataset == 'luad':
                    image_label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
                elif self.dataset == 'bcss':
                    image_label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])
                path_label.append((image_path, image_label))
        return path_label
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.GaussianBlur(21),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                 std=[0.229, 0.224, 0.225]),
            ])
        return composed_transforms(sample)

class Stage2_Dataset(Dataset):
    def __init__(self, base_dir, split):

        super().__init__()
        self._base_dir = base_dir
        self.split = split

        if self.split == 'val':
            self._image_dir = os.path.join(self._base_dir, 'LUAD-HistoSeg/val/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'LUAD-HistoSeg/val/mask/')
        elif self.split == 'test':
            self._image_dir = os.path.join(self._base_dir, 'LUAD-HistoSeg/test/img/')
            self._cat_dir   = os.path.join(self._base_dir, 'LUAD-HistoSeg/test/mask/')
        self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self._image_dir) if not file.startswith('.')]
        self.images = [os.path.join(self._image_dir, fn + '.png') for fn in self.filenames]
        self.categories = [os.path.join(self._cat_dir, fn + '.png') for fn in self.filenames]

        assert (len(self.images) == len(self.categories))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if (self.split == 'val') or (self.split == 'test'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]

        if (self.split == 'val') or (self.split == 'test'):
            return self.transform_val(sample), image_dir
        
    def _make_img_gt_point_pair(self, index):
        if (self.split == 'val') or (self.split == 'test'):
            _img = Image.open(self.images[index]).convert('RGB')
            _target = Image.open(self.categories[index])
            return _img,_target

    # def transform_tr_ab(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip_ab(),
    #         tr.RandomGaussianBlur_ab(),
    #         tr.Normalize_ab(),
    #         tr.ToTensor_ab()])
    #     return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return None

# def make_data_loader(args, **kwargs):

#     train_set   = Stage1_TrainDataset(args, base_dir=args.dataroot, transform=, dataset='luad')
#     val_set     = Stage2_Dataset(args, base_dir=args.dataroot, split='val')
#     test_set    = Stage2_Dataset(args, base_dir=args.dataroot, split='test')

#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#     test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

#     return train_loader, val_loader, test_loader
