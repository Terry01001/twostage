import os
import torch
import torch.utils.data as data
from .dataset import IncrementalSegmentationDataset
import numpy as np
from . import transform
# from torchvision import transforms as tr
from torch import from_numpy

from PIL import Image

# classes = {
#     0: 'background',
#     1: 'Tumor epithelial (TE)',
#     2: 'Necrosis (NEC)',
#     3: 'Lymphocyte (LYM)',
#     4: 'Tumor-associated stroma (TAS)' 
# }

class LUADSegmentation(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 transform=None,
                 step=0,
                 masking_value=4,
                 masking=True,):

    # def __init__(self,
    #             root,
    #             train=True,
    #             transform=None,
    #             indices=None,
    #             as_coco=False,
    #             saliency=False,
    #             pseudo=None):
        
        self.data_path = root
        base_dir = "luad"
        self.data_path = os.path.join(self.data_path, base_dir)
        self.file = []
        if train:
            self.split = 'train'
            self.data_path = os.path.join(self.data_path, 'training')
            self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self.data_path) if not file.startswith('.')]
            for f in self.filenames:
                fname = f
                label_str = fname.split(']')[0].split('[')[-1]
                image_label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
                
                if image_label[2] == 1 or image_label[1] == 1: # at least one object label in the image 
                    # self.file.append(f)
                    pass
                else: 
                    self.file.append(f)
                    # pass
            self.images = [os.path.join(self.data_path, fn + '.png') for fn in self.file]
        else:
            self.split = 'val'
            self.data_path = os.path.join(self.data_path, 'val')
            self._image_dir = os.path.join(self.data_path, 'img')
            self._cat_dir   = os.path.join(self.data_path, 'mask')
            self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self._image_dir) if not file.startswith('.')]
            self.images = [os.path.join(self._image_dir, fn + '.png') for fn in self.filenames]
            self.categories = [os.path.join(self._cat_dir, fn + '.png') for fn in self.filenames]
            #self.indices = np.arange(len(self.images))
            assert (len(self.images) == len(self.categories))

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step
        self.transform = transform

        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        # print(self.order)

        if step > 0:
            # self.labels = [self.order[0]] + list(step_dict[step])
            self.labels = list(step_dict[step])
        else:
            self.labels = list(step_dict[step])
        self.labels_old = [lbl for s in range(step) for lbl in step_dict[s]]

        # print(self.labels)
        # print(self.labels_old)



        self.masking_value = masking_value
        self.masking = masking
        
        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        # print(self.inverted_order) 
        

        if train:
            self.inverted_order[4] = masking_value
        else:
            self.set_up_void_test()

        # print(self.inverted_order) 
        # print(self.labels.asdasd())

        if masking:
            tmp_labels = self.labels + [4]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order
        
        # print(mapping_dict)
        # print(self.labels.asdasd())

        # train = False
        # step += 1
        # if not train:
        #     mapping = torch.zeros((5,))
        #     mapping += 4
        #     for k in mapping_dict.keys():
        #         mapping[k] = mapping_dict[k]
            
        #     if step > 0:
        #         for i in self.labels_old:
        #             mapping[i] = self.labels_old[i]
        #     self.transform_lbl = LabelTransform(mapping)
        # else:
        #     self.dataset.img_lvl_only = True
        # print(mapping)
        # print(self.order.sdsadasd())
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)

        if train:
            self.object = self.path_label()
        self.indices = np.arange(len(self.images))
        
    def set_up_void_test(self):
        self.inverted_order[4] = 4

    def __getitem__(self, index):
        if self.split == 'train':
            fn, label = self.object[self.indices[index]]
            # print(label)
            img = Image.open(fn).convert('RGB')
            # label = label[self.indices[index]]
            if self.transform is not None:
                img = self.transform(img)
            label = self.transform_1h(label)
            img, img_box = transform.random_crop(
                    img,
                    crop_size=224,
                    mean_rgb=[0,0,0],#[123.675, 116.28, 103.53], 
                    ignore_index=4)
        
            # label = label[1:3]
            # print("after")
            # print(label)
            # print(label.dsds())
            return img, label, img_box
        else:
            img, label = self._make_img_gt_point_pair(index)
            # sample = {'image': _img, 'label': _target}
            # image_dir = self.images[index]
            if self.transform is not None:
                img, label = self.transform(img, label)
            # print(type(label))
            # label = self.transform_lbl(label)
            return img, label
        
    def __len__(self):
        return len(self.images)
        
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
                image_label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
                if image_label[2] == 1 or image_label[1] == 1:
                    pass
                    # path_label.append((image_path, image_label))
                else: 
                    path_label.append((image_path, image_label))
                    # pass

        # self.indices = np.arange(len(path_label))
        return path_label

    def _make_img_gt_point_pair(self, index):

        _img = Image.open(self.images[self.indices[index]]).convert('RGB')
        _target = Image.open(self.categories[self.indices[index]])

        return _img,_target
    


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        # print(x)
        for k in x:
            if x[int(k)] not in self.mapping:
                x[int(k)] = self.mapping
        # print(x)
        # return from_numpy(self.mapping[x])
        return x


class LabelSelection:
    def __init__(self, order, labels, masking):
        # print(order)
        order = np.array(order)
        # print(order)
        # order = order[order != 0]
        # print(order)
        # order -= 1  # scale to match one-hot index.
        # print(order)
        self.order = order
        masking = True
        if masking:
            self.masker = np.zeros((len(order)))
            self.masker[-len(labels):] = 1
        else:
            self.masker = np.ones((len(order)))

        # print(self.masker)
        # print(order.sdsd())

    def __call__(self, x):
        # print(x)
        x = x[self.order] * self.masker
        # print(x)
        # print(x.sdadasd())
        return x


class LUADSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = LUADSegmentation(root, train, transform=None)
        return full_voc