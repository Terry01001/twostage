import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import gm
import torchvision
from tool.wss_loss import *
from module.modules import ASPP


def train(path_work, teacher_model, model, dataloader_train, device, hp, valid_fn=None, dataloader_valid=None, test_num_pos=0):
    if hp['pretrain'] is True:
        teacher_model = teacher_model.pretrain(teacher_model, device)
        model = model.pretrain(model, device)
    
    r = hp['r']
    lr = hp['lr']
    wd = hp['wd']
    num_epoch = hp['epoch']
    best_mIoU = 0
    best_FWIoU = 0
    best_IoU = 0
    best_ACC = 0
    print('Learning Rate: ', lr)
    loss_fn = nn.BCELoss()
    dataset_size = len(dataloader_train.dataset)
    # aspp = ASPP(384).to(device)
    param_groups = teacher_model.get_parameter_groups()

    # optimizer = torchutils.PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
    #     {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
    #     {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    # ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if hp['optimizer'] == 'side':
        params1 = list(map(id, model.decoder1.parameters()))
        params2 = list(map(id, model.decoder2.parameters()))
        params3 = list(map(id, model.decoder3.parameters()))
        base_params = filter(lambda p: id(p) not in params1 + params2 + params3, model.parameters())
        params = [{'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
                  {'params': param_groups[1], 'lr': 2*lr, 'weight_decay': 0},
                  {'params': param_groups[2], 'lr': 10*lr, 'weight_decay': wd},
                  {'params': param_groups[3], 'lr': 20*lr, 'weight_decay': 0},
                  {'params': base_params},
                  {'params': model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                #   {'params': aspp.parameters(), 'lr': lr / 100, 'weight_decay': wd}
                ]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    print("{:*^50}".format("training start"))
    for epoch in range(num_epoch):
        teacher_model.train()
        model.train()
        epoch_loss = 0
        step = 0
        batch_num = len(dataloader_train)

        for index, batch in enumerate(dataloader_train):
            _, image, label = batch
            image = image.to(device)
            label = label.to(device)
            
            x, feature, y = teacher_model(image, enable_PDA=True)
            d3, side1, side2, side3, fusion = model(image)

            loss_t = F.multilabel_soft_margin_loss(y, label)
            # print(x.shape)
            # print(feature.shape)
            # print(y.shape)

            # print(fusion.shape)
            # print(x.sdasdasdasd())
            # aspp.eval()
            # pseudolbl_mask = aspp(d3).detach() #torch.Size([32, 4, 14, 14])
            # print(pseudolbl.shape)
            # print(pseudolbl.sdsdsd())

            # aspp.train()
            # pseudolbl = aspp(d3)

            # loss_lbl = bce_loss(pseudolbl, label, mode='ngwp', reduction='none')

            # pseudolbl_mask =nn.functional.adaptive_max_pool2d(pseudolbl_mask,(1,1))
            # pseudolbl_mask = pseudolbl_mask.view(image.shape[0], label.shape[1], -1).sum(-1)
            
            
            side1 =nn.functional.adaptive_max_pool2d(side1,(1,1))
            side1 = side1.view(image.shape[0], label.shape[1], -1).sum(-1)
            side2 =nn.functional.adaptive_max_pool2d(side2,(1,1))
            side2 = side2.view(image.shape[0], label.shape[1], -1).sum(-1)
            side3 =nn.functional.adaptive_max_pool2d(side3,(1,1))
            side3 = side3.view(image.shape[0], label.shape[1], -1).sum(-1)

            # lossf2 = bce_loss(fusion, label, mode='ngwp', reduction='none')
            
            fusion =nn.functional.adaptive_max_pool2d(fusion,(1,1))
            fusion = fusion.view(image.shape[0], label.shape[1], -1).sum(-1)
            # print(fusion.shape)
            # print(d3.sdsdsdas())
            loss1 = loss_fn(side1, label)
            loss2 = loss_fn(side2, label)
            loss3 = loss_fn(side3, label)
            lossf = loss_fn(fusion, label)

            ms = torch.softmax(fusion, dim=1)
            loss_ms = loss1 + loss2 + loss3 + lossf
            # teacher_outputs.argmax(dim=1)
            loss_kd = F.cross_entropy(ms, x.argmax(dim=1))
            # loss_kd = F.cross_entropy(x, ms.long())
            # loss_pseudo = loss_fn(fusion, pseudolbl_mask)
            # loss = loss1 + loss2 + loss3 + lossf
            # loss1 = bce_loss(side1, label, mode='ngwp', reduction='none')
            # loss2 = bce_loss(side2, label, mode='ngwp', reduction='none')
            # loss3 = bce_loss(side3, label, mode='ngwp', reduction='none')
            # lossf2 = bce_loss(fusion, label, mode='ngwp', reduction='none')
            loss = loss_ms + loss_t + loss_kd
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1
            
            if index % 10 == 0:
                print("batch %d/%d loss:%0.4f" % (index, batch_num, loss.item()))
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
        print("epoch %d loss:%0.4f" % (epochs, average_loss))

        if valid_fn is not None:
            model.eval()
            result = valid_fn(model, dataloader_valid, test_num_pos, device)
            print('Validation:')
            print('epoch %d loss:%.4f' % (epochs, average_loss))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(result['Acc'], result['Acc_class'], result['mIoU'], result['FWIoU']))
            print('IoUs: ', result['ious'])

            if result['mIoU'] > best_mIoU:
                best_mIoU = result['mIoU']
                best_ACC = result['Acc']
                best_FWIoU = result['FWIoU']
                best_IoU = result['ious']

                torch.save(model.state_dict(), path_work + 'best_model.pth')
        torch.save(model.state_dict(), path_work + 'final_model.pth')

    # print('best result: %.3f' % best_mIoU)
    print("Best result => IoUs:{}, fwIoU:{}, mIoU:{}, Acc:{}".format(best_IoU, best_FWIoU, best_mIoU, best_ACC))

