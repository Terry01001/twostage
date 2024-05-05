import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils import gm
from functools import reduce
import torchvision
from tool.wss_loss import *
from tool.dist_loss import DistillationLoss
from tool.focal_loss import focal_loss
from tool.scheduler import get_scheduler
from module.modules import ASPP, PAMR
from module.single_stage import pseudo_gtmask, balanced_mask_loss_ce, balanced_mask_loss_unce, _rescale_and_clean
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tool.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_aff_mat,propagte_aff_cam,
                            propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)
from tool.loss import FocalLoss
from kornia.geometry.transform import flips
from module.PAR import PAR
from tool import imutils
from tool.losses import KDLoss, softmax_kl_loss, softmax_mse_loss
import tasks
from models import *
from tqdm import tqdm

global_step = 0

torch.autograd.set_detect_anomaly(True)

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w 
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask

def get_seg_loss(pred, label, ignore_index=4):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

def get_aff_loss(inputs, targets, device):
    # print(targets.shape)
    upsp = torch.nn.Upsample(scale_factor=4)
    c1 = nn.Conv1d(49, 196, 1)
    c1 = c1.to(device)
    # inputs = F.interpolate(inputs, size=(196,196), mode='linear', align_corners=False)
    inputs = upsp(inputs)
    inputs = c1(inputs)
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)
    # print(pos_label.shape)
    # print(inputs.shape)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # print(model.parameters())
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        if 'backbone' in param[0]:  # 假設 encoder 參數名稱中包含 'encoder'
            ema_param[1].data.mul_(alpha).add_(1 - alpha, param[1].data)
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def batch_rotation(grid, rots):
    ret = []
    for i, rot in enumerate(rots):
        ret.append(grid[i, ...].rot90(-int(rot // 90), [1,2]))
    return torch.stack(ret, 0)


def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=0.45] = 0

    return valid_cam, _pseudo_label

def multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        
        cam_list = [F.relu(_cam)]

        # for s in scales:
        #     if s != 1.0:
        #         _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
        #         inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

        #         _cam, _ = model(inputs_cat, cam_only=True)

        #         _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #         _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

        #         cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam

def train(opts, path_work, model, t_model, ema_model, model_old, dataloader_train, device, hp, valid_fn=None, pseudo_valid_fn=None, dataloader_valid=None, test_num_pos=0, batch_size=64):
    # if hp['pretrain'] is True:
        # teacher_model = teacher_model.pretrain(teacher_model, device)
    global global_step
    if opts.step == 0:
        if opts.backbone == 'SwinUNETR' :
            try:
                for w in model.modules():
                    if isinstance(w, nn.Conv2d):
                        nn.init.xavier_uniform_(w.weight)

                model_dict = model.state_dict()
                pretrained_dict = torch.load('pretrained_models/swin_tiny_patch4_window7_224.pth')['model']
                new_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if 'backbone.' + k in model_dict.keys()}
                model_dict.update(new_dict)
                model.load_state_dict(model_dict)
                print("Using pretrained Swin backbone weights !")
            except ValueError:
                raise ValueError("pre-trained weights not available for" + str(opts.backbone))
        else:
            model = model.pretrain(opts, model, device)
            # t_model = t_model.pretrain(opts, t_model, device)
        

    

    if ema_model is not None:  
        ema_model.to(device)
        # freeze ema model and set eval mode
        for para in ema_model.parameters():
            para.requires_grad = False
        ema_model.eval()

    if model_old is not None:  # if step 0, we don't need to instance the model_old
        model_old.to(device)
        # freeze old model and set eval mode
        for para in model_old.parameters():
            para.requires_grad = False
        model_old.eval()


    
    classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
    if classes is not None:
        new_classes = classes[-1]
        tot_classes = reduce(lambda a, b: a + b, classes)
        old_classes = tot_classes - new_classes
    else:
        old_classes = 0


    r = hp['r']
    lr = hp['lr']
    wd = hp['wd']
    num_epoch = hp['epoch']
    best_mIoU = 0
    best_FWIoU = 0
    best_IoU = 0
    best_ACC = 0
    # attn_mask = get_mask_by_radius(h=14, w=14, radius=8)
    # attn_mask_infer = get_mask_by_radius(h=7, w=7, radius=8)
    # par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
    # par.to(device)
    # bkg_cls = torch.ones(size=(batch_size, 1))
    use_aff = opts.affinity
    if use_aff:
        affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
        # affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
        # affinity = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).to(device)
        for p in affinity.parameters():
            p.requires_grad = False

    print('Learning Rate: ', lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_ce = nn.CrossEntropyLoss(ignore_index=4, reduction='none')
    # loss_fl = focal_loss(alpha=[1,2,2,1], gamma=2, num_classes=4)
    loss_fl = FocalLoss(ignore_index=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_dl = nn.KLDivLoss()
    kd_loss = KDLoss(T=10)

    if opts.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif opts.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, opts.consistency_type

    dataset_size = len(dataloader_train.dataset)

    pseudolabeler = None
    if opts.step > 0:
        channels = 384
        aspp = ASPP(512)
        # pseudolabeler = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                         nn.BatchNorm2d(256),
        #                                         nn.ReLU(256),
        #                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                         nn.BatchNorm2d(256),
        #                                         nn.ReLU(256),
        #                                         nn.Conv2d(256, 2, kernel_size=1, stride=1))
        pseudolabeler = aspp
        pseudolabeler.to(device)

    if hp['optimizer'] == 'side':
        if opts.stage == 'two' and opts.phase == 0: 
            params1 = list(map(id, model.decoder1.parameters()))
            params2 = list(map(id, model.decoder2.parameters()))
            params3 = list(map(id, model.decoder3.parameters()))
            # t_params3 = list(map(id, t_model.decoder3.parameters()))
            # params4 = list(map(id, model.decoder4.parameters()))
            refine_params = list(map(id, model.refine_module.parameters()))
            base_params = filter(lambda p: id(p) not in params1 + params2 + params3 + refine_params , model.parameters()) # params3 + params4
            # t_base_params = filter(lambda p: id(p) not in t_params3, t_model.parameters())
            params = [
                    {'params': base_params, 'lr': lr, 'weight_decay': wd},
                      {'params': model.decoder1.parameters(), 'lr': lr/100, 'weight_decay': wd},
                      {'params': model.decoder2.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                      {'params': model.decoder3.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                    #   {'params': model.decoder4.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                      {'params': model.refine_module.parameters(), 'lr': lr/100 , 'weight_decay': wd}
                    ]
            # t_params = [
            #         {'params': t_base_params, 'lr': lr, 'weight_decay': wd},
            #         #   {'params': model.decoder1.parameters(), 'lr': lr/100, 'weight_decay': wd},
            #         #   {'params': model.decoder2.parameters(), 'lr': lr/100 , 'weight_decay': wd},
            #           {'params': t_model.decoder3.parameters(), 'lr': lr/100 , 'weight_decay': wd},
            #         #   {'params': model.decoder4.parameters(), 'lr': lr/100 , 'weight_decay': wd},
            #         ]
            if opts.step > 0:
                params.append({"params": filter(lambda p: p.requires_grad, pseudolabeler.parameters()),
                            'lr': opts.lr_pseudo, 'weight_decay': opts.weight_decay})
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            # t_optimizer = torch.optim.AdamW(t_params, lr=lr, weight_decay=wd)
            # optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
            scheduler = get_scheduler(opts, optimizer)
            # t_scheduler = get_scheduler(opts, t_optimizer)

        elif opts.stage == 'two' and opts.phase == 1:
            params4 = list(map(id, model.decoder.parameters()))
            base_params = filter(lambda p: id(p) not in params4, model.parameters())
            params = [
                    {'params': base_params, 'lr': lr, 'weight_decay': wd},
                    {'params': model.decoder.parameters(), 'lr': lr/10 , 'weight_decay': wd},
                    ]
            if opts.step > 0:
                params.append({"params": filter(lambda p: p.requires_grad, pseudolabeler.parameters()),
                            'lr': opts.lr_pseudo, 'weight_decay': opts.weight_decay})
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            # optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
            scheduler = get_scheduler(opts, optimizer)
        
        else:
            # params1 = list(map(id, model.decoder1.parameters()))
            params2 = list(map(id, model.decoder2.parameters()))
            params3 = list(map(id, model.decoder3.parameters()))
            paramsd4 = list(map(id, model.decoder4.parameters()))
            params4 = list(map(id, model.decoder.parameters()))
            base_params = filter(lambda p: id(p) not in params2 + params3 + paramsd4 + params4, model.parameters())
            params = [
                    {'params': base_params, 'lr': lr, 'weight_decay': wd},
                    #   {'params': model.decoder1.parameters(), 'lr': lr/100, 'weight_decay': wd},
                      {'params': model.decoder2.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                      {'params': model.decoder3.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                      {'params': model.decoder4.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                      {'params': model.decoder.parameters(), 'lr': lr/10 , 'weight_decay': wd},
                    ]
            if opts.step > 0:
                params.append({"params": filter(lambda p: p.requires_grad, pseudolabeler.parameters()),
                            'lr': opts.lr_pseudo, 'weight_decay': opts.weight_decay})
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            scheduler = get_scheduler(opts, optimizer)

    else:
        model_params = [v for k, v in model.named_parameters() if 'out' not in k]
        class_params = model.out.parameters()
        optimizer = torch.optim.AdamW(
            [{'params': class_params, 'lr': lr/10},
            {'params': model_params, 'lr': 10*lr}],
            lr=lr, weight_decay=wd)
        scheduler = get_scheduler(opts, optimizer)
    
    print("{:*^50}".format("training start"))
    for epoch in range(num_epoch):
        model.train()
        # t_model.train()
        epoch_loss = 0
        # loss = 0
        step = 0
        batch_num = len(dataloader_train)
        num_updates = epoch * batch_num
        epoch_loss = 0.0
        reg_loss = 0.0
        l_cam_out = 0.0
        l_cam_int = 0.0
        l_seg = 0.0
        l_cls = 0.0
        interval_loss = 0.0

        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        # print('Learning Rate: ', optimizer.param_groups[0]['lr'])
        loop = tqdm((dataloader_train), total=len(dataloader_train))

        for index, batch in enumerate(loop):
            if opts.dataset == 'WSSS4LUAD':
                image, label, img_box = batch
                image = image.to(device)
                label = label.to(device=device, dtype=torch.float)
            else:
                image, strong_image, label, img_box = batch
                image = image.to(device)
                strong_image = strong_image.to(device)
                label = label.to(device=device, dtype=torch.float)

            loss = 0


            if model_old is not None:
                with torch.no_grad():
                    side1_old, side2_old, side3_old, fusion_old, attns_old, d3_old, msf_old = model_old(image)

            if ema_model is not None:
                with torch.no_grad():
                    s2, s3, s4, pseudo_gt = ema_model(image)

            if opts.backbone == 'SwinUNETR':
                output = model(image)
            elif opts.stage == 'two' and opts.phase == 0:
                # side1, side2, side3, fusion = model(image)
                # with torch.no_grad():
                #     t_fusion = t_model(image)
                side1, side2, side3, side4, fusion = model(image)
                
                bs, c, h, w = fusion.size()

            elif opts.stage == 'two' and opts.phase == 1:
                #side2, side3, fusion, msf= model(image)
                cls4, side2, side3, side4, fusion, msf= model(image)
                bs, c, h, w = msf.size()
                # fusion, attns, d3, msf = model(image)
            else:
                cls4, side2, side3, side4, fusion, msf= model(image)
                bs, c, h, w = msf.size()

            # inputs_denorm = imutils.denormalize_img2(image.clone())



            ################## CAM #######################
            # if epoch > 0:
            # cams, aff_mat = multi_scale_cam_with_aff_mat(model, inputs=image, scales=[1, 0.5, 1.5])

            # pseudo_label = cam_to_label(cams.detach(), cls_label=label) #torch.Size([64, 224, 224])

            # valid_cam_resized = F.interpolate(cams, size=(aff_mat.shape[-1], aff_mat.shape[-1]), mode='bilinear', align_corners=False)

            # aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=label, bkg_score=0.35)
            # aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            # aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=label, bkg_score=0.55)
            # aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

            # bkg_cls = bkg_cls.to(cams.device)                    # path_label.append((image_path, image_label))
            # _cls_labels = torch.cat((bkg_cls, label), dim=1)

            # refined_aff_label_l = aff_cam_l.argmax(dim=1)
            # refined_aff_label_h = aff_cam_h.argmax(dim=1)

            # aff_cam = aff_cam_l[:,1:]
            # refined_aff_cam = aff_cam_l[:,1:,]
            # refined_aff_label = refined_aff_label_h.clone()

            
            # refined_aff_label[refined_aff_label_h == 4] = 4
            # refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 4] = 4


            ######################with img_box##################################
            # cams, aff_mat = multi_scale_cam_with_aff_mat(model, inputs=image, scales=[1, 0.5, 1.5])
            # valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=label, img_box=img_box, ignore_mid=False)
            # # print(valid_cam.shape)
            # # print(pseudo_label.shape)
            # # print(cams.sdsdasd())

            # # pseudo_label = cam_to_label(cams.detach(), cls_label=label) #torch.Size([64, 224, 224])

            # valid_cam_resized = F.interpolate(valid_cam, size=(aff_mat.shape[-1], aff_mat.shape[-1]), mode='bilinear', align_corners=False)

            # aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=label, bkg_score=0.35)
            # aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            # aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=label, bkg_score=0.55)
            # aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

            # bkg_cls = bkg_cls.to(cams.device)
            # _cls_labels = torch.cat((label, bkg_cls), dim=1)

            # refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box)
            # refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
            # refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box)
            # refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

            # # refined_aff_label_l = aff_cam_l.argmax(dim=1)
            # # refined_aff_label_h = aff_cam_h.argmax(dim=1)

            # # aff_cam = aff_cam_l[:,1:]
            # # refined_aff_cam = aff_cam_l[:,1:,]
            # # refined_aff_label = refined_aff_label_h.clone()

            
            # # refined_aff_label[refined_aff_label_h == 4] = 4
            # # refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 4] = 4          
            # aff_cam = aff_cam_l[:,1:]
            # refined_aff_cam = refined_aff_cam_l[:,1:,]
            # refined_aff_label = refined_aff_label_h.clone()
            # refined_aff_label[refined_aff_label_h == 4] = 4
            # refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 4] = 4
            # refined_aff_label = ignore_img_box(refined_aff_label, img_box=img_box, ignore_index=4)  

            # refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=label, img_box=img_box)

            # aff_label = cams_to_affinity_label(refined_pseudo_label, mask=attn_mask, ignore_index=4)
            # aff_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label, device)

            # fusion = F.interpolate(fusion, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

            # if epoch <= 5:
            #     refined_aff_label = refined_pseudo_label

            # seg_loss = get_seg_loss(segs, refined_aff_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
            # cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)
            ####################################################################

            ##############################
            # loss_seg = get_seg_loss(fusion, refined_aff_label.type(torch.long), ignore_index=4)
            # cls_loss = F.multilabel_soft_margin_loss(cls, label)
            ##############################


            if opts.step > 0:
                bs = image.shape[0]

                pseudolabeler.eval()
                int_masks = pseudolabeler(msf).detach()

                pseudolabeler.train()
                int_masks_raw = pseudolabeler(msf)

                # l_cam_new = bce_loss(int_masks_raw, label, mode='ngwp', reduction='mean')
                y = ngwp_focal(int_masks_raw) #[32, 2]
                l_cam_new = loss_fn(torch.sigmoid(y), label)
                # int_masks_raw = F.interpolate(int_masks_raw, size=fusion_old.shape[-1], mode='bilinear', align_corners=False)

                # l_loc = F.binary_cross_entropy_with_logits(int_masks_raw[:, :old_classes],
                #                                             torch.sigmoid(fusion_old.detach()),
                #                                             reduction='mean')
                l_cam_int = l_cam_new

                if epoch >= 5:

                    int_masks_orig = int_masks.softmax(dim=1)
                    int_masks_soft = int_masks.softmax(dim=1)

                    if use_aff:
                        image_raw = image
                        im = F.interpolate(image_raw, int_masks.shape[-2:], mode="bilinear",
                                            align_corners=True)
                        int_masks_soft = affinity(im, int_masks_soft.detach())
                    
                    int_masks_orig[:, :] *= label[:, :,None,None]
                    int_masks_soft[:, :] *= label[:, :,None,None]

                    pseudo_gt_seg = pseudo_gtmask(int_masks_soft, ambiguous=True, cutoff_top=0.6,
                                                    cutoff_bkg=0.7, cutoff_low=0.2).detach()  # B x C x HW

                    pseudo_gt_seg_lx = binarize(int_masks_orig)
                    pseudo_gt_seg_lx = (opts.alpha * pseudo_gt_seg_lx) + \
                                        ((1-opts.alpha) * int_masks_orig)

                    # ignore_mask = (pseudo_gt_seg.sum(1) > 0)
                    px_cls_per_image = pseudo_gt_seg_lx.view(bs, tot_classes, -1).sum(dim=-1)
                    batch_weight = torch.eq((px_cls_per_image[:, old_classes:] > 0),
                                            label[:, old_classes:].bool())
                    batch_weight = (
                                batch_weight.sum(dim=1) == (tot_classes - old_classes)).float()

                    target_old = torch.sigmoid(fusion_old.detach())
                    pseudo_gt_seg_lx = F.interpolate(pseudo_gt_seg_lx, size=fusion_old.shape[-1], mode='bilinear', align_corners=False)

                    target = torch.cat((target_old, pseudo_gt_seg_lx[:, old_classes:]), dim=1)
                    # if opts.icarl_bkg == -1:
                    #     target[:, 0] = torch.min(target[:, 0], pseudo_gt_seg_lx[:, 0])
                    # else:
                    #     target[:, 0] = (1-opts.icarl_bkg) * target[:, 0] + \
                    #                     opts.icarl_bkg * pseudo_gt_seg_lx[:, 0]

                    l_seg = F.binary_cross_entropy_with_logits(fusion, target, reduction='none').sum(dim=1)
                    l_seg = l_seg.view(bs, -1).mean(dim=-1)
                    l_seg = opts.l_seg * (batch_weight * l_seg).sum() / (batch_weight.sum() + 1e-5)

                    l_cls = balanced_mask_loss_ce(fusion, pseudo_gt_seg, label)

                l_sc = l_seg 
                l_reg = l_cls + l_cam_int
                loss += l_sc + l_reg
            else:
                if opts.backbone == 'SwinUNETR':
                    bs, c, h, w = output.size()
                    # y = ngwp_focal(output) #[32, 4]
                    # loss += loss_fn(torch.sigmoid(y), label)
                    masks = F.softmax(output, dim=1)
                    masks_ = masks.view(bs, c, -1)
                    y =nn.functional.adaptive_avg_pool2d(output,(1,1))
                    y = y.view(image.shape[0], label.shape[1], -1).sum(-1)
                    y_focal = torch.pow(1 - masks_.mean(-1), 3) * torch.log(0.01 + masks_.mean(-1))
                    y = y + y_focal

                    loss += loss_fn(torch.sigmoid(y), label)
                else:
                    if opts.stage == 'two' and opts.phase == 0:
                        # side1 =nn.functional.adaptive_max_pool2d(side1,(1,1))
                        # side1 = side1.view(image.shape[0], label.shape[1], -1).sum(-1)
                        side2 =nn.functional.adaptive_max_pool2d(side2,(1,1))
                        side2 = side2.view(image.shape[0], label.shape[1], -1).sum(-1)
                        side3 =nn.functional.adaptive_max_pool2d(side3,(1,1))
                        side3 = side3.view(image.shape[0], label.shape[1], -1).sum(-1)
                        side4 =nn.functional.adaptive_max_pool2d(side4,(1,1))
                        side4 = side4.view(image.shape[0], label.shape[1], -1).sum(-1)

                        # rotations = np.random.choice([0, 90, 180, 270], fusion.shape[0], replace=True)
                        # images = flips.Hflip()(image)
                        # images_rotated = batch_rotation(images, rotations)
                        # _l2, _l3, logits_rotated = model(images_rotated)
                        # logits_recovered = batch_rotation(logits_rotated, 360 - rotations)
                        # logits_recovered = flips.Hflip()(logits_recovered)
                        
                        # flip_loss = torch.mean(torch.abs(logits_recovered-fusion))

                        # _l2, _l3, _l4, logits_flip = model(flips.Hflip()(image))
                        # flip_loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-fusion))

                        fusion_ =nn.functional.adaptive_max_pool2d(fusion,(1,1))
                        fusion_ = fusion_.view(image.shape[0], label.shape[1], -1).sum(-1)
                        # t_fusion_ =nn.functional.adaptive_max_pool2d(t_fusion,(1,1))
                        # t_fusion_ = t_fusion_.view(image.shape[0], label.shape[1], -1).sum(-1)

                        # pseudo = F.softmax(t_fusion, dim=1)
                        # l_seg = F.binary_cross_entropy_with_logits(fusion, t_fusion)
                        # loss1 = loss_fn(side1, label)
                        loss2 = loss_fn(side2, label)  #(N,C)?
                        loss3 = loss_fn(side3, label)
                        loss4 = loss_fn(side4, label)
                        lossf = loss_fn(fusion_, label)
                        # loss_ms = loss1 + loss2 + loss3 + lossf
                        # print(loss2.item(),loss3.item(),loss4.item(),lossf.item())
                        loss_ms = loss2 +  loss3 +  loss4 + lossf
                        loss += loss_ms

                    elif opts.stage == 'two' and opts.phase == 1:
                        msf_ = ngwp_focal(msf)

                        # if epoch >= 5:
                        #     int_masks = fusion.detach()

                        #     int_masks_orig = int_masks.softmax(dim=1)
                        #     int_masks_soft = int_masks.softmax(dim=1)

                        #     if use_aff:
                        #         image_raw = image
                        #         im = F.interpolate(image_raw, int_masks.shape[-2:], mode="bilinear",
                        #                             align_corners=True)
                        #         int_masks_soft = affinity(im, int_masks_soft.detach())

                        #     int_masks_orig[:, :] = int_masks_orig[:, :] * label[:, :,None,None]
                        #     int_masks_soft[:, :] = int_masks_soft[:, :] * label[:, :,None,None]

                        #     pseudo_gt_seg = pseudo_gtmask(int_masks_soft, ambiguous=True, cutoff_top=0.6,
                        #                                     cutoff_bkg=0.7, cutoff_low=0.2).detach()  # B x C x HW

                        #     pseudo_gt_seg_lx = binarize(int_masks_orig)
                        #     pseudo_gt_seg_lx = (opts.alpha * pseudo_gt_seg_lx) + \
                        #                         ((1-opts.alpha) * int_masks_orig)
                            
                        #     pseudo_gt_seg_lx = F.interpolate(pseudo_gt_seg_lx, size=msf.shape[-1], mode='bilinear', align_corners=False)
                        #     target = pseudo_gt_seg_lx.clone()

                        #     l_seg = F.binary_cross_entropy_with_logits(msf, target)

                        #     l_cls = balanced_mask_loss_ce(fusion, pseudo_gt_seg, label)
                        
                        # pseudo_gt_orig = F.softmax(pseudo_gt, dim=1)
                        # pseudo_gt = F.softmax(pseudo_gt, dim=1)
                        pseudo_gt = torch.argmax(pseudo_gt, dim=1)
                        if use_aff:
                            image_raw = image
                            im = F.interpolate(image_raw, pseudo_gt.shape[-2:], mode="bilinear",
                                                align_corners=True)
                            pseudo_gt = affinity(im, pseudo_gt.detach())

                        # loss_seg = loss_fn(torch.sigmoid(msf_), label)
                        msf = F.interpolate(msf, size=pseudo_gt.shape[-1], mode='bilinear', align_corners=False)

                        # l_seg = F.binary_cross_entropy_with_logits(msf, pseudo_gt)

                        # pseudo_gt = torch.argmax(pseudo_gt, dim=1)
                        
                        # pseudo_gt = pseudo_gt.squeeze(1)
                        # msf = torch.argmax(msf, dim=1)
                        # print(msf.shape)
                        # print(pseudo_gt.shape)
                        # pseudo_gt = torch.tensor(pseudo_gt, dtype=torch.int64, device=device)
                        # msf = torch.tensor(msf, dtype=torch.int64, device=device)

                        l_seg = F.cross_entropy(msf, pseudo_gt.long())
                        
                        # online easy example mining 

                        # normalized loss
                        # weight = torch.ones_like(l_seg)
                        # # print(l_seg)
                        # metric = -l_seg.detach().reshape((l_seg.shape[0], l_seg.shape[1] * l_seg.shape[2]))
                        # weight = F.softmax(metric, 1)
                        # weight = weight / weight.mean(1).reshape((-1, 1))
                        # weight = weight.reshape((l_seg.shape[0], l_seg.shape[1], l_seg.shape[2]))
                        
                        # # apply oeem on images of multiple labels
                        # for i in range(label.shape[0]):
                        #     tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist()) - {255}
                        #     if len(tag) <= 1:
                        #         weight[i] = 1
                        
                        # reduction='mean'
                        # avg_factor=None
                        # if weight is not None:
                        #     weight = weight.float()
                        # l_seg = weight_reduce_loss(
                        #     l_seg, weight=weight, reduction=reduction, avg_factor=avg_factor)
                        
                        loss +=  l_seg
                    else:
                        side2_ =nn.functional.adaptive_max_pool2d(side2,(1,1))
                        side2_ = side2_.view(image.shape[0], label.shape[1], -1).sum(-1)
                        side3_ =nn.functional.adaptive_max_pool2d(side3,(1,1))
                        side3_ = side3_.view(image.shape[0], label.shape[1], -1).sum(-1)
                        side4_ =nn.functional.adaptive_max_pool2d(side4,(1,1))
                        side4_ = side4_.view(image.shape[0], label.shape[1], -1).sum(-1)
                        fusion_ =nn.functional.adaptive_max_pool2d(fusion,(1,1))
                        fusion_ = fusion_.view(image.shape[0], label.shape[1], -1).sum(-1)
                        loss2 = loss_fn(side2_, label)
                        loss3 = loss_fn(side3_, label)
                        loss4 = loss_fn(side4_, label)
                        lossf = loss_fn(fusion_, label)
                        loss_ms = loss2+loss3+loss4+lossf

                        # cams = multi_scale_cam(model, image, 1)
                        # valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=label)
                        # print(valid_cam.shape) #B*C*H*W
                        # print(pseudo_label.shape) #B*H*W
                        # print(cams.sdadasdasd())

                        # y = ngwp_focal(msf)
                        # l_cam_new = loss_fn(torch.sigmoid(y), label)
                        
                        # side2_flip, side3_flip, f_flip, logits_flip = model(flips.Hflip()(image))
                        # flip_loss = torch.mean(torch.abs(flips.Hflip()(f_flip)-fusion))
                        # side2_soft = torch.softmax(side2, dim=1)
                        # side3_soft = torch.softmax(side3, dim=1)
                        # cross_loss1 = kd_loss(side2,side3.detach()) + kd_loss(side2,side4.detach())
                        # cross_loss2 = kd_loss(side3,side2.detach()) + kd_loss(side3,side4.detach())
                        # cross_loss3 = kd_loss(side4,side2.detach()) + kd_loss(side4,side3.detach())
                        # cross_consist = (cross_loss1 + cross_loss2 + cross_loss3)/3
                        # cross_loss = 0.1*cross_consist





                        if epoch >= 10:
                            # pseudo_gt = F.softmax(fusion.detach(), dim=1)
                            cams = multi_scale_cam(model, image, 1)
                            # cam_loss = torch.mean(torch.abs(fusion-cams.detach().float()))
                            # pseudo_gt = torch.argmax(fusion.detach(), dim=1)
                            pseudo_gt = F.softmax(fusion.detach(), dim=1)
                            # pseudo_gt = fusion.detach()
                            if use_aff:
                                image_raw = image
                                im = F.interpolate(image_raw, cams.shape[-2:], mode="bilinear",
                                                    align_corners=True)
                                cams = affinity(im, cams)
                            
                            cam_loss = torch.mean(torch.abs(fusion-cams.detach().float()))
                            # cams = multi_scale_cam(model, image, 1)
                            # valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=label)
                            
                            # cam_loss = torch.mean(torch.abs(pseudo_gt-pseudo_label.float()))

                            # pseudo_gt[:, :] = pseudo_gt[:, :] * label[:, :,None,None]
                            # pseudo_gt = pseudo_gt.detach()
                            
                            # pseudo_gt_seg_lx = binarize(pseudo_gt)
                            # pseudo_gt_seg_lx = (opts.alpha * pseudo_gt_seg_lx) + \
                            #                     ((1-opts.alpha) * pseudo_gt)
                            # pseudo_gt_seg_lx = pseudo_gtmask(pseudo_gt, ambiguous=True, cutoff_top=0.6,
                            #                                 cutoff_bkg=0.7, cutoff_low=0.2).detach()  # B x C x HW

                            # loss_seg = loss_fn(torch.sigmoid(msf_), label)
                            msf = F.interpolate(msf, size=pseudo_gt.shape[-1], mode='bilinear', align_corners=False)
                            # side2_flip, side3_flip, f_flip, logits_flip = model(flips.Hflip()(image))
                            # msf = F.interpolate(msf, size=side3.shape[-1], mode='bilinear', align_corners=False)
                            # flip_loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-msf))
                            # pseudo_gt = torch.argmax(pseudo_gt, dim=1)
                            # weight = torch.tensor([1, 5 ,18, 1], dtype=torch.float, device=device)
                            l_seg = F.binary_cross_entropy_with_logits(msf, pseudo_gt)
                            # l_seg = F.cross_entropy(msf, pseudo_gt.long(), reduction='mean')
                            # online easy example mining 

                            # normalized loss
                            # weight = torch.ones_like(l_seg)
                            # metric = -l_seg.detach().reshape((l_seg.shape[0], l_seg.shape[1] * l_seg.shape[2]))
                            # weight = F.softmax(metric, 1)
                            # weight = weight / weight.mean(1).reshape((-1, 1))
                            # weight = weight.reshape((l_seg.shape[0], l_seg.shape[1], l_seg.shape[2]))
 
                            # apply oeem on images of multiple labels
                            # for i in range(label.shape[0]):
                            #     tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist()) - {255}
                            #     if len(tag) <= 1:
                            #         weight[i] = 1
                            
                            # reduction='mean'
                            # avg_factor=None
                            # if weight is not None:
                            #     weight = weight.float()
                            # l_seg = weight_reduce_loss(
                            #     l_seg, weight=weight, reduction=reduction, avg_factor=avg_factor)
    

                            # _l2, _l3, logits_flip, _msf = model(flips.Hflip()(image))
                            # flip_loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-fusion))
                            
                            loss = loss + l_seg + cam_loss

                        # else:
                        # y = ngwp_focal(cls4)
                        l_cam_new = loss_fn(torch.sigmoid(cls4), label)
                        loss = loss + l_cam_new

                        loss += loss_ms



            optimizer.zero_grad()
            # t_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # t_optimizer.step()
            if opts.lr_policy == 'polyLR':
                num_updates += 1
                scheduler.step_update(num_updates=num_updates)
                # t_scheduler.step_update(num_updates=num_updates)
            else:
                scheduler.step()
                # t_scheduler.step()

            global_step += 1
            # if epoch >= 5:
            #     update_ema_variables(model, ema_model, opts.ema_decay, global_step)
            

            epoch_loss += loss.item()
            step += 1
            loop.set_description('Epoch [%d/%d], lr:%0.8f' % (epoch+1, num_epoch, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=loss.item())

            # if index % 10 == 0:
            #     print("batch %d/%d loss:%0.4f" % (index, batch_num, loss.item()))
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
        # print("epoch %d loss:%0.4f" % (epochs, average_loss))

        if valid_fn is not None and epochs % opts.val_interval == 0:
            model.eval()
            if pseudolabeler is not None:
                pseudolabeler.eval()
            result = valid_fn(model, pseudolabeler,  dataloader_valid, test_num_pos, device)
            print('Validation:')
            print('epoch %d loss:%.4f' % (epochs, average_loss))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(result['Acc'], result['Acc_class'], result['mIoU'], result['FWIoU']))
            print('IoUs: ', result['ious'])

            if opts.stage == 'one' or opts.phase == 1:
                pseudo_result = pseudo_valid_fn(model, pseudolabeler,  dataloader_valid, test_num_pos, device)
                print('Pseudo Validation:')
                # print('epoch %d loss:%.4f' % (epochs, average_loss))
                print("Pseudo_Acc:{}, Pseudo_Acc_class:{}, Pseudo_mIoU:{}, Pseudo_fwIoU: {}".format(pseudo_result['Pseudo_Acc'], pseudo_result['Pseudo_Acc_class'], pseudo_result['Pseudo_mIoU'], pseudo_result['Pseudo_FWIoU']))
                print('Pseudo_IoUs: ', pseudo_result['Pseudo_ious'])

            if result['mIoU'] > best_mIoU:
                best_mIoU = result['mIoU']
                best_ACC = result['Acc']
                best_FWIoU = result['FWIoU']
                best_IoU = result['ious']

                if opts.phase == 0:
                    torch.save(model.state_dict(), path_work + f's{opts.step}_{opts.dataset}_p{opts.phase}_best_model.pth')
                else:
                    torch.save(model.state_dict(), path_work + f's{opts.step}_{opts.dataset}_p{opts.phase}_best_model.pth')
        if opts.phase == 0:
            torch.save(model.state_dict(), path_work + f's{opts.step}_{opts.dataset}_p{opts.phase}_final_model.pth')
        else:
            torch.save(model.state_dict(), path_work + f's{opts.step}_{opts.dataset}_p{opts.phase}_final_model.pth')
        # torch.save(model.state_dict(), path_work + f's{opts.step}_p2_final_model.pth')

    # print('best result: %.3f' % best_mIoU)
    print("Best result => IoUs:{}, fwIoU:{}, mIoU:{}, Acc:{}".format(best_IoU, best_FWIoU, best_mIoU, best_ACC))

