import os
import torch.utils.data as data
from torch import from_numpy
import numpy as np


class IncrementalSegmentationDataset(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 masking_value=4,
                 step=0,
                 weakly=False,
                 pseudo=None):

        # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
        if train:
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path)
            # else:
            #     raise FileNotFoundError(f"Please, add the traning spilt in {idxs_path}.")
        else:  # In both test and validation we want to use all data available (even if some images are all bkg)
            idxs = None

        # self.dataset = self.make_dataset(root, train, indices=idxs, pseudo=pseudo)
        self.dataset = self.make_dataset(root, train, indices=None, pseudo=pseudo)
        self.transform = transform
        self.weakly = weakly  # don't use weakly in val
        self.train = train

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step

        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        # print(self.order)
        
        # assert not any(l in labels_old for l in labels), "Labels and labels_old must be disjoint sets"
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
        # print(self.masking_value)



        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        # print(self.inverted_order)

        if train:
            self.inverted_order[4] = masking_value
        else:
            self.set_up_void_test()

        # print(self.inverted_order)
        

        if masking:
            tmp_labels = self.labels + [4]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order
        
        # print(mapping_dict)

        # train = False
        if not train:
            mapping = np.zeros((256,))
            mapping += 4
            for k in mapping_dict.keys():
                mapping[k] = mapping_dict[k]
            
            if step > 0:
                for i in self.labels_old:
                    mapping[i] = self.labels_old[i]
            self.transform_lbl = LabelTransform(mapping)
        # else:
        #     self.dataset.img_lvl_only = True
        # print(mapping)
        # print(self.order.sdsadasd())
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)

    def set_up_void_test(self):
        self.inverted_order[4] = 4

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        if index < len(self):
            if self.train:
                data = self.dataset[index]
                # img, lbl, lbl_1h = data[0], data[1], data[2]
                img, lbl_1h = data[0], data[1]
                img =  self.transform(img)
                lbl_1h = self.transform_1h(lbl_1h)
                return img, lbl_1h
            else:
                data = self.dataset[index]
                # img, lbl, lbl_1h = data[0], data[1], data[2]
                img, lbl = data[0], data[1]
                img, lbl = self.transform(img, lbl)
                # print(lbl)
                # print(lbl.sadadad())
                lbl = self.transform_lbl(lbl)
                return img, lbl

        else:
            raise ValueError("absolute value of index should not exceed dataset length")

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        raise NotImplementedError


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])


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
        # masking = True
        if masking:
            self.masker = np.zeros((len(order)))
            self.masker[-len(labels):] = 1
        else:
            self.masker = np.ones((len(order)))

        # print(self.masker)
        # print(order.sdsd())

    def __call__(self, x):
        x = x[self.order] * self.masker
        # print(x)
        # print(x.sdadasd())
        return x
