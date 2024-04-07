import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageFilter


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms[index]

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self):
        return len(self.transforms)

    def __call__(self, img, lbl=None):
        if lbl is not None:
            for tr in self.transforms:
                img, lbl = tr(img, lbl)
                #print(img)
                #print(img.asdad())
            return img, lbl
        else:
            for tr in self.transforms:
                img = tr(img)
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string



class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        img = image
        mask = label
        # print(mask.shape)
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        # print(mask.shape)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img, mask
    
class Normalize_w(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        if lbl is not None:
            # img = tensor
            # mask = lbl
            # # print(mask.shape)
            # img = np.array(img).astype(np.float32)
            # mask = np.array(mask).astype(np.float32)
            # # print(mask.shape)
            tensor /= 255.0
            # img -= self.mean
            # img /= self.std
            return F.normalize(tensor, self.mean, self.std), lbl
        else:
            # img = tensor
            # # print(mask.shape)
            # img = np.array(img).astype(np.float32)
            # # print(mask.shape)
            tensor /= 255.0
            # img -= self.mean
            # img /= self.std
            return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, label):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = image
        mask = label

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return img, mask

class RandomHorizontalFlip(object):
    def __call__(self, image, label):
        img = image
        mask = label
        # print(mask)
        # print(type(mask))
        # print(mask.shape)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask
    
class RandomVerticalFlip(object):
    def __call__(self, image, label):
        img = image
        mask = label

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return img, mask

class RandomGaussianBlur(object):
    def __call__(self, image, label):
        img = image
        mask = label
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img, mask
                

class Normalize_ab(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask_a = sample['label_a']
        mask_b = sample['label_b']
        # print(mask.shape)
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        mask_a = np.array(mask_a).astype(np.float32)
        mask_b = np.array(mask_b).astype(np.float32)
        # print(mask.shape)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask,
                'label_a': mask_a,
                'label_b': mask_b}

class ToTensor_ab(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        mask_a = sample['label_a']
        mask_b = sample['label_b']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        mask_a = np.array(mask_a).astype(np.float32)
        mask_b = np.array(mask_b).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        mask_a = torch.from_numpy(mask_a).float()
        mask_b = torch.from_numpy(mask_b).float()

        return {'image': img,
                'label': mask,
                'label_a': mask_a,
                'label_b': mask_b}



class RandomHorizontalFlip_ab(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask_a = sample['label_a']
        mask_b = sample['label_b']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask_a = mask_a.transpose(Image.FLIP_LEFT_RIGHT)
            mask_b = mask_b.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img,
                'label': mask,
                'label_a': mask_a,
                'label_b': mask_b}

class RandomGaussianBlur_ab(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask_a = sample['label_a']
        mask_b = sample['label_b']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask,
                'label_a': mask_a,
                'label_b': mask_b}



class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}





class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}