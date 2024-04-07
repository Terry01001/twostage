import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tool.metrics import Evaluator, Evaluator_BCSS


def valid(model, pseudolabeler, dataloader, test_num_pos, device):
    step = 0
    num = 0
    total_f = 0
    nclass = 4
    evaluator = Evaluator(nclass)
    evaluator.reset()
    # c1 = torch.nn.Conv2d(4096, 4, 1)
    # c1 = c1.to(device)
    # c1 = nn.Sequential(
    #       nn.Conv2d(4096,2048,1),
    #       nn.ReLU(),
    #       nn.Conv2d(2048,1024,1),
    #       nn.ReLU(),
    #       nn.Conv2d(1024,4,1),
    #       nn.ReLU(),
    #     )
    # c1 = c1.to(device)
    # F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic') 
    # torch.set_printoptions(profile="full")
    

    with torch.no_grad():
        for sample in dataloader:
            image, label = sample      
            # print(image.shape)
            # print(label.shape)
            # print(image.asdadad())
            preds = model(image.to(device))

            
            # pred = torch.argmax(preds[3], dim=1).cpu().numpy() #mit_b4, swin_t
            # pred = torch.argmax(preds, dim=1).cpu().numpy() #SwinUNETR

            pred = F.interpolate(preds[-1], size=(label.shape[-1], label.shape[-1]), mode='bilinear', align_corners=False)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            # torch.set_printoptions(profile="full")
            # np.set_printoptions(edgeitems=25088)
            # print(pred)
            # print(pred.sadas())
            

            target = label.squeeze(1).cpu().numpy()



            # for i in range(4):
            #     if i in pred:
            #         print("pred ", i)
            #     if i in target:
            #         print("target ", i)
            # print(pred.shape)
            # print(target.shape)
            # print(pred.asdad())


            #step 0 : class 0, 3 
            # pred[pred==1]=5
            # pred[pred==3]=1
            # pred[pred==5]=3
            # pred[pred==1]=4
            # pred[pred==2]=4



            # #step 1 : class 1, 2 
            # pred[pred==0]=5
            # pred[pred==2]=0
            # pred[pred==5]=2

            # torch.set_printoptions(profile="full")
            # np.set_printoptions(edgeitems=25088)

            ### LUAD-HistoSeg and BCSS-WSSS
            ## cls 4 is exclude
            pred[target==4]=4

            # mask = np.ones_like(target)
            # target = target - mask
            # target[target==-1]=3

            # pred[target==3]=3
            # print(target.shape)
            # print(pred.shape)
            # print(pred.sdasdasdasd())



            evaluator.add_batch(target, pred)

            # pred = ((preds[3] >= 0.5) + 0).squeeze(0).squeeze(0).to('cpu').numpy()
            # label = label.squeeze(0).squeeze(0).int().to("cpu").numpy()

            # temp1 = label.sum().item()
            # if temp1 != 0:
            #     f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
            # else:
            #     f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)

            # total_f += f1
        # average_f = total_f/num
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        ious = evaluator.Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        return {'Acc':Acc, 'Acc_class':Acc_class, 'mIoU':mIoU, 'ious':ious, 'FWIoU':FWIoU}