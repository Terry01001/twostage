from .voc import VOCSegmentation, VOCSegmentationIncremental, VOCasCOCOSegmentationIncremental
from .coco import COCO, COCOIncremental
from .luad import LUADSegmentation, LUADSegmentationIncremental
# from .transform import *
from dataset import custom_transforms as tr
from dataset import transforms as tf
from torchvision import transforms as transform
import tasks
import os


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    # train_transform = transform.Compose([
    #     transform.RandomResizedCrop(opts.crop_size, (0.5, 2)),
    #     transform.RandomHorizontalFlip(),
    #     transform.ToTensor(),
    #     transform.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    # ])
    # val_transform = transform.Compose([
    #     transform.Resize(size=opts.crop_size_val),
    #     transform.CenterCrop(size=opts.crop_size_val),
    #     transform.ToTensor(),
    #     transform.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    train_transform = tf.Compose([
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        tf.RandomGaussianBlur(),
        tf.ToTensor(),
        tf.Normalize(),
        # transform.Normalize(mean=[0., 0., 0.],
        #                     std=[0.229, 0.224, 0.225]),
    ])
    val_transform = tf.Compose([
        # transform.Resize(size=opts.crop_size_val),
        # transform.CenterCrop(size=opts.crop_size_val),
        # tr.RandomHorizontalFlip(),
        # tr.RandomVerticalFlip(),
        # tr.RandomGaussianBlur(),
        tf.ToTensor(),
        tf.Normalize(),
    ])

    test_transform = val_transform

    step_dict = tasks.get_task_dict(opts.dataset, opts.task, opts.step)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    # print(step_dict)
    # print(labels)
    # print(labels_old)
    # print(path_base)
    # print(labels.ssdadad())

    pseudo = f"{opts.pseudo}_{opts.task}_{opts.step}" if opts.pseudo is not None else None
    path_base = os.path.join(opts.data_root, path_base)

    labels_cum = labels_old + labels
    # masking_value = 0
    masking_value = 4

    if opts.dataset == 'voc':
        t_dataset = dataset = VOCSegmentationIncremental
    elif opts.dataset == 'coco':
        t_dataset = dataset = COCOIncremental
    elif opts.dataset == 'coco-voc':
        if opts.step == 0:
            t_dataset = dataset = COCOIncremental
        else:
            dataset = VOCasCOCOSegmentationIncremental
            t_dataset = COCOIncremental
    elif opts.dataset == 'luad':
        t_dataset = dataset = LUADSegmentation
    else:
        raise NotImplementedError 

    if opts.overlap and opts.dataset == 'voc':
        path_base += "-ov"

    path_base_train = path_base

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    # print(opts.dataset)

    if opts.dataset == 'luad':
        train_dst = dataset(root=opts.data_root, step_dict=step_dict, train=True, transform=train_transform, 
                            masking_value=masking_value, masking=opts.no_mask, step=opts.step)

        # Val is masked with 0 when label is not known or is old (masking=True, masking_value=0)
        val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform, 
                          masking_value=masking_value, masking=not opts.no_mask, step=opts.step)

        # Test is masked with 255 for labels not known and the class for old (masking=False, masking_value=255)
        image_set = 'train' if opts.val_on_trainset else 'val'
        test_dst = t_dataset(root=opts.data_root, step_dict=step_dict, train=opts.val_on_trainset, transform=test_transform,
                             masking_value=masking_value,masking=not opts.no_mask, step=opts.step)
        # train_dst = dataset(root=opts.data_roo, step_dict=step_dict, train=True, transform=train_transform,
        #                     masking_value=masking_value,
        #                     masking=not opts.no_mask, overlap=opts.overlap, step=opts.step, weakly=opts.weakly,
        #                     pseudo=pseudo)


        # # Val is masked with 0 when label is not known or is old (masking=True, masking_value=0)
        # val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform,
        #                 masking_value=masking_value,
        #                 masking=False, overlap=opts.overlap, step=opts.step, weakly=opts.weakly)

        # # Test is masked with 255 for labels not known and the class for old (masking=False, masking_value=255)
        # image_set = 'train' if opts.val_on_trainset else 'val'
        # test_dst = t_dataset(root=opts.data_root, step_dict=step_dict, train=opts.val_on_trainset, transform=test_transform,
        #                     masking=False, masking_value=255, weakly=opts.weakly,
        #                     step=opts.step)
    else:
        train_dst = dataset(root=opts.data_root, step_dict=step_dict, train=True, transform=train_transform,
                            idxs_path=path_base_train + f"/train-{opts.step}.npy", masking_value=masking_value,
                            masking=not opts.no_mask, overlap=opts.overlap, step=opts.step, weakly=opts.weakly,
                            pseudo=pseudo)


        # Val is masked with 0 when label is not known or is old (masking=True, masking_value=0)
        val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform,
                        idxs_path=path_base + f"/val-{opts.step}.npy", masking_value=masking_value,
                        masking=False, overlap=opts.overlap, step=opts.step, weakly=opts.weakly)

        # Test is masked with 255 for labels not known and the class for old (masking=False, masking_value=255)
        image_set = 'train' if opts.val_on_trainset else 'val'
        test_dst = t_dataset(root=opts.data_root, step_dict=step_dict, train=opts.val_on_trainset, transform=test_transform,
                            masking=False, masking_value=255, weakly=opts.weakly,
                            idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy", step=opts.step)

    return train_dst, val_dst, test_dst, labels_cum, len(labels_cum)

