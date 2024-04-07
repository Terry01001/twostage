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
from module.PAR import PAR
from tool import imutils
from tool import losses, ramps
import tasks
from models import *
from tqdm import tqdm

global_step = 0


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

def train(opts, path_work, model, ema_model, model_old, dataloader_train, device, hp, valid_fn=None, dataloader_valid=None, test_num_pos=0, batch_size=64):
    # if hp['pretrain'] is True:
        # teacher_model = teacher_model.pretrain(teacher_model, device)
    global global_step
    if opts.step == 0:
        if opts.backbone == 'SwinUNETR' :
            try:
                for w in model.modules():
                    if isinstance(w, nn.Conv2d):
                        nn.init.xavier_uniform_(w.weight)
            #     model_dict = torch.load("./pretrained_models/swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt")
            #     state_dict = model_dict["state_dict"]
            #     # fix potential differences in state dict keys from pre-training to
            #     # fine-tuning
            #     if "module." in list(state_dict.keys())[0]:
            #         print("Tag 'module.' found in state dict - fixing!")
            #         for key in list(state_dict.keys()):
            #             state_dict[key.replace("module.", "")] = state_dict.pop(key)
            #     if "swin_vit" in list(state_dict.keys())[0]:
            #         print("Tag 'swin_vit' found in state dict - fixing!")
            #         for key in list(state_dict.keys()):
            #             state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            #     # We now load model weights, setting param `strict` to False, i.e.:
            #     # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            #     # the decoder weights untouched (CNN UNet decoder).
            #     model.load_state_dict(state_dict, strict=False)
            #     print("Using pretrained self-supervised Swin UNETR backbone weights !")
            # except ValueError:
            #     raise ValueError("Self-supervised pre-trained weights not available for" + str(opts.backbone))

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
        
        ema_model = ema_model.pretrain(opts, ema_model, device)
        # for param in ema_model.parameters():
        #         param.detach_()
    
    # model.to(device)
    
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
    # print(classes)
    # print(new_classes)
    # print(tot_classes)
    # print(old_classes)
    # print(old_classes.sdsdasd())


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
        # affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
        affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
        for p in affinity.parameters():
            p.requires_grad = False

    print('Learning Rate: ', lr)
    loss_fn = nn.BCELoss()
    loss_ce = nn.CrossEntropyLoss(ignore_index=4, reduction='none')
    # loss_fl = focal_loss(alpha=[1,2,2,1], gamma=2, num_classes=4)
    loss_fl = FocalLoss(ignore_index=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_dl = nn.KLDivLoss()

    if opts.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif opts.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
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
        params1 = list(map(id, ema_model.decoder1.parameters()))
        params2 = list(map(id, ema_model.decoder2.parameters()))
        params3 = list(map(id, ema_model.decoder3.parameters()))
        params4 = list(map(id, model.decoder.parameters()))
        # params4 = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in params4, model.parameters())
        base_params_ema = filter(lambda p: id(p) not in params1+params2+params3, ema_model.parameters())
        params = [
                #   {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
                #   {'params': param_groups[1], 'lr': 2*lr, 'weight_decay': 0},
                #   {'params': param_groups[2], 'lr': 10*lr, 'weight_decay': wd},
                #   {'params': param_groups[3], 'lr': 20*lr, 'weight_decay': 0},
                  {'params': base_params, 'lr': lr, 'weight_decay': wd},
                #   {'params': model.decoder1.parameters(), 'lr': lr/100, 'weight_decay': wd},
                #   {'params': model.decoder2.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                #   {'params': model.decoder3.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                  {'params': model.decoder.parameters(), 'lr': lr/10 , 'weight_decay': wd},
                #   {'params': model.classifier.parameters(), 'lr': lr , 'weight_decay': wd},
                #   {'params': aspp.parameters(), 'lr': lr / 100, 'weight_decay': wd}
                ]
        ema_params = [
                #   {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
                #   {'params': param_groups[1], 'lr': 2*lr, 'weight_decay': 0},
                #   {'params': param_groups[2], 'lr': 10*lr, 'weight_decay': wd},
                #   {'params': param_groups[3], 'lr': 20*lr, 'weight_decay': 0},
                  {'params': base_params_ema, 'lr': lr, 'weight_decay': wd},
                  {'params': ema_model.decoder1.parameters(), 'lr': lr/100, 'weight_decay': wd},
                  {'params': ema_model.decoder2.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                  {'params': ema_model.decoder3.parameters(), 'lr': lr/100 , 'weight_decay': wd},
                #   {'params': model.decoder.parameters(), 'lr': lr/10 , 'weight_decay': wd},
                #   {'params': model.classifier.parameters(), 'lr': lr , 'weight_decay': wd},
                #   {'params': aspp.parameters(), 'lr': lr / 100, 'weight_decay': wd}
                ]
        if opts.step > 0:
            params.append({"params": filter(lambda p: p.requires_grad, pseudolabeler.parameters()),
                          'lr': opts.lr_pseudo, 'weight_decay': opts.weight_decay})
        # optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        optimizer_ema = torch.optim.AdamW(ema_params, lr=lr, weight_decay=wd)
        # optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        # scheduler = get_scheduler(opts, optimizer)
        scheduler_ema = get_scheduler(opts, optimizer_ema)
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model_params = [v for k, v in model.named_parameters() if 'out' not in k]
        class_params = model.out.parameters()
        # print(model.named_para)
        # print(class_params)
        optimizer = torch.optim.AdamW(
            [{'params': class_params, 'lr': lr/10},
            {'params': model_params, 'lr': 10*lr}],
            lr=lr, weight_decay=wd)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        scheduler = get_scheduler(opts, optimizer)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    
    print("{:*^50}".format("training start"))
    for epoch in range(num_epoch):
        # teacher_model.train()
        # if epoch>=20:
        #     for param in ema_model.parameters():
        #         param.detach_()
        # else:
        for param in ema_model.parameters():
            param.requires_grad = True
        model.train()
        ema_model.train()
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
            image, label, img_box = batch
            image = image.to(device)
            label = label.to(device=device, dtype=torch.float)
            loss = 0
            minibatch_size = len(label)


            if model_old is not None:
                with torch.no_grad():
                    side1_old, side2_old, side3_old, fusion_old, attns_old, d3_old, msf_old = model_old(image)

            if opts.backbone == 'SwinUNETR':
                output = model(image)
            else:
                # msf = model(image)
                side1, side2, side3, fusion = ema_model(image)
                # fusion, attns, d3, msf = model(image)

            inputs_denorm = imutils.denormalize_img2(image.clone())

            bs, c, h, w = fusion.size()



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
                    side1 =nn.functional.adaptive_max_pool2d(side1,(1,1))
                    side1 = side1.view(image.shape[0], label.shape[1], -1).sum(-1)
                    side2 =nn.functional.adaptive_max_pool2d(side2,(1,1))
                    side2 = side2.view(image.shape[0], label.shape[1], -1).sum(-1)
                    side3 =nn.functional.adaptive_max_pool2d(side3,(1,1))
                    side3 = side3.view(image.shape[0], label.shape[1], -1).sum(-1)

                    fusion_ =nn.functional.adaptive_max_pool2d(fusion,(1,1))
                    fusion_ = fusion_.view(image.shape[0], label.shape[1], -1).sum(-1)
                    # fusion_ = ngwp_focal(fusion)


                    # msf_ =nn.functional.adaptive_max_pool2d(msf,(1,1))
                    # msf_ = msf_.view(image.shape[0], label.shape[1], -1).sum(-1)
                    # msf_ = ngwp_focal(msf)

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

                    loss1 = loss_fn(side1, label)
                    loss2 = loss_fn(side2, label)
                    loss3 = loss_fn(side3, label)
                    lossf = loss_fn(fusion_, label)
                    # loss_seg = loss_fn(torch.sigmoid(msf_), label)
                    loss_ms = loss1 + loss2 + loss3 + lossf
                    # loss_ms = loss_seg
                    # loss_ms = loss_fn(torch.sigmoid(msf_), label)
                    # if opts.consistency and epoch >= 20:
                    #     consistency_weight = get_current_consistency_weight(opts, epoch)
                    #     consistency_loss = consistency_weight * consistency_criterion(fusion_, msf_) / minibatch_size
                    # else:
                    #     consistency_loss = 0

                    # loss += loss_seg + consistency_loss
                    loss += loss_ms

            # loss += loss_ms + 0.1*aff_loss + 0.1*loss_seg + cls_loss
            # loss = loss.mean()
            # if epoch >= 20:
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     if opts.lr_policy == 'polyLR':
            #         num_updates += 1
            #         scheduler.step_update(num_updates=num_updates)
            #     else:
            #         scheduler.step()
            #     global_step += 1
            #     update_ema_variables(model, ema_model, opts.ema_decay, global_step)
            # else:
            #     # optimizer.zero_grad()
            #     optimizer_ema.zero_grad()
            #     # loss.backward()
            #     loss_ms.backward()
            #     # optimizer.step()
            #     optimizer_ema.step()
            #     if opts.lr_policy == 'polyLR':
            #         num_updates += 1
            #         scheduler.step_update(num_updates=num_updates)
            #     else:
            #         scheduler.step()
            #     global_step += 1
                    
            # optimizer.zero_grad()
            optimizer_ema.zero_grad()
            loss.backward()
            # loss_ms.backward()
            # optimizer.step()
            optimizer_ema.step()
            if opts.lr_policy == 'polyLR':
                num_updates += 1
                scheduler_ema.step_update(num_updates=num_updates)
            else:
                scheduler_ema.step()
            global_step += 1

            epoch_loss += loss.item()
            step += 1
            loop.set_description('Epoch [%d/%d], lr:%0.8f' % (epoch+1, num_epoch, optimizer_ema.param_groups[0]['lr']))
            # loop.set_description('Epoch [%d/%d], lr:%0.8f' % (epoch+1, num_epoch, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=loss.item())

            # if index % 10 == 0:
            #     print("batch %d/%d loss:%0.4f" % (index, batch_num, loss.item()))
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
        # print("epoch %d loss:%0.4f" % (epochs, average_loss))

        if valid_fn is not None and epochs % opts.val_interval == 0:
            # model.eval()
            ema_model.eval()
            if pseudolabeler is not None:
                pseudolabeler.eval()
            result = valid_fn(ema_model, pseudolabeler,  dataloader_valid, test_num_pos, device)
            print('Validation:')
            print('epoch %d loss:%.4f' % (epochs, average_loss))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(result['Acc'], result['Acc_class'], result['mIoU'], result['FWIoU']))
            print('IoUs: ', result['ious'])

            if result['mIoU'] > best_mIoU:
                best_mIoU = result['mIoU']
                best_ACC = result['Acc']
                best_FWIoU = result['FWIoU']
                best_IoU = result['ious']

                torch.save(ema_model.state_dict(), path_work + f's{opts.step}_ema_best_model.pth')
        torch.save(ema_model.state_dict(), path_work + f's{opts.step}_final_model.pth')

    # print('best result: %.3f' % best_mIoU)
    print("Best result => IoUs:{}, fwIoU:{}, mIoU:{}, Acc:{}".format(best_IoU, best_FWIoU, best_mIoU, best_ACC))


def get_current_consistency_weight(opts, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return opts.consistency * ramps.sigmoid_rampup(epoch, opts.consistency_rampup)

