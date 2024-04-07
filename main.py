import argparser
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import os
import importlib
from timm.models import create_model
from functools import reduce
from monai.networks.nets import SwinUNETR

from models import *
from functions import *
from datasets_wsss import Dataset_train, Dataset_valid, Dataset_test
from dataset_w4l import OriginPatchesDataset, w4l_valid
from utils import *
from tool.GenDataset import *
from tool.mixup import Mixup
from dataset import get_dataset
import tasks


def main(opts):
    print('Loading......')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_name = opts.dataset
    
    path_work = 'work/test/'
    if os.path.exists(path_work) is False:
        os.mkdir(path_work)

    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataroot = 'data/'
    classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
    # print(classes)
    if classes is not None:
        new_classes = classes[-1]
        tot_classes = reduce(lambda a, b: a + b, classes)
        old_classes = tot_classes - new_classes
    else:
        old_classes = 0

    # print(new_classes)
    # print(tot_classes)
    # print(old_classes)
        
    # CUTMIX CONFIGURATION
    cutmix_fn = None
    # cutmix_fn = Mixup(mixup_alpha=0, cutmix_alpha=1,
    #                     cutmix_minmax=[0.4, 0.8], prob=1, switch_prob=0, 
    #                     mode="single", correct_lam=True, label_smoothing=0.0,
    #                     num_classes=opts.num_classes)
    batch_size = 32

    if dataset_name == 'WSSS4LUAD':
        data_path_name = f'data/{dataset_name}/1.training'
        TrainDataset = OriginPatchesDataset(opts.val_on_trainset, data_path_name=data_path_name, 
                                            cutmix_fn=cutmix_fn, num_class=opts.num_classes)
        TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
        dataloader_train = DataLoader(TrainDataset, batch_size=batch_size, num_workers=4, sampler=TrainDatasampler, drop_last=True)

        dataset_size = [224, 224]
        dataset_valid = w4l_valid(dataset_size, device, dataset_name)
        dataloader_valid = DataLoader(dataset_valid, batch_size=8, shuffle=False, num_workers=4)

    else:
        dataset_size = [224, 224]
        dataset_train = Dataset_train(dataset_size, device, dataset_name, opts.val_on_trainset , opts.step)
        dataset_valid = Dataset_valid(dataset_size, device, dataset_name)
        dataset_test = Dataset_test(dataset_size, device, dataset_name)

        # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        # dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=8)
        # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size if not opts.val_on_trainset else 1, shuffle=True, num_workers=4, drop_last=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=4)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)
    test_num_pos = 80
    opts.max_iters = opts.epochs * len(dataloader_train)

    # if opts.backbone == 'SwinUNETR':
    #     model = SwinUNETR(img_size=(224,224), 
    #                       in_channels=3, 
    #                       out_channels=tot_classes, 
    #                       use_checkpoint=False, 
    #                       spatial_dims=2).to(device)
    # else:
    #     model = Swin_MIL(opts, classes=tot_classes, ema=False).to(device)

    ema_model = None
    t_model = None
    if opts.stage == 'two':
        if opts.phase == 0:
            model = Swin_MIL(opts, classes=tot_classes, ema=True).to(device)
            # t_model = Swin_MIL(opts, classes=tot_classes, ema=True).to(device)
        else:
            model = Swin_MIL(opts, classes=tot_classes, ema=False).to(device)
            ema_model = Swin_MIL(opts, classes=tot_classes, ema=True).to(device)
            if opts.step_ckpt is not None:
                path = opts.step_ckpt
            else:
                path = f'work/test/s{opts.step}_{opts.dataset}_p0_best_model.pth'
            step_checkpoint = torch.load(path, map_location="cpu")
            # ema_model = Swin_MIL(opts, classes=tot_classes, ema=True)
            ema_model.load_state_dict(step_checkpoint, strict=True) 
    else:
        model = Swin_MIL(opts, classes=tot_classes, ema=False).to(device)
    
    # ema_model = Swin_MIL(opts, classes=tot_classes, ema=True).to(device)
    
    # if opts.step_ckpt is not None:
    #     path = opts.step_ckpt
    # else:
        # path = f'work/test/s{opts.step}_ema_best_model.pth'
    # trainer.load_step_ckpt(path)
    # step_checkpoint = torch.load(path, map_location="cpu")
    # new_dict = {'backbone.' + k: v for k, v in step_checkpoint.items() if 'backbone.' + k in model.state_dict().keys()}
    # model.state_dict().update(new_dict)


    # ema_model = Swin_MIL(opts, classes=tot_classes, ema=True)
    # ema_model.load_state_dict(step_checkpoint, strict=True) 

    # print(model)
    # print(model.sadasdas())
    model_old = None

    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f'work/test/s{opts.step - 1}_best_model.pth'
        # trainer.load_step_ckpt(path)
        step_checkpoint = torch.load(path, map_location="cpu")
        # new_dict = {'backbone.' + k: v for k, v in step_checkpoint.items() if 'backbone.' + k in model.state_dict().keys()}
        # model.state_dict().update(new_dict)


        model_old = Swin_MIL(opts, classes=2)
        model_old.load_state_dict(step_checkpoint, strict=True) 

        step_checkpoint.pop('decoder1.0.weight')
        step_checkpoint.pop('decoder1.0.bias')
        step_checkpoint.pop('decoder2.0.weight')
        step_checkpoint.pop('decoder2.0.bias')
        step_checkpoint.pop('decoder3.0.weight')
        step_checkpoint.pop('decoder3.0.bias')
        model.load_state_dict(step_checkpoint, strict=False)  # False for incr. classifiers
        # if opts.init_balanced:
        #     # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
        #     model.module.init_new_classifier(device)
        # Load state dict from the model state dict, that contains the old model parameters
        # new_state = {}
        # for k, v in step_checkpoint.items():
        #     new_state[k[9:]] = v
 # Load also here old parameters

        # clean memory
        del step_checkpoint
        # new_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if 'backbone.' + k in model_dict.keys()}
        # model_dict.update(new_dict)
        # model_dict.pop('head.weight')
        # model_dict.pop('head.bias')
        # model.load_state_dict(model_dict)

    # teacher_model = teacher_model.to(device)
    

    hyperparameters = {
        'r' : 4,
        'lr' : 1e-4,# 1e-4, # 1e-5
        'wd' : 0.0005, #0.0005
        'epoch' : 20,
        'pretrain' : True,
        'optimizer' : 'side'  # side
    }

    print('Dataset: ' + dataset_name)
    print('Data Volume: ', len(dataloader_train.dataset))
    # print('Teacher Model: ', type(teacher_model))
    print('Model: ', type(model))
    print('Backbone:', opts.backbone)
    print('Batch Size: ', batch_size)
    print('Epochs:', hyperparameters['epoch'])
    train(opts, path_work, model, t_model, ema_model, model_old, dataloader_train, device, hyperparameters, valid, valid_pseudo, dataloader_valid, test_num_pos, batch_size)
    # test(path_work, model, dataloader_test, device)


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("checkpoints/step", exist_ok=True)
    main(opts)





