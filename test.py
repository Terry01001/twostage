import argparser
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# from datasets_onelabel import Dataset_train_onelabel, Dataset_valid_onlabel, Dataset_test_onelabel
from utils import *
from tool.GenDataset import *
from tool.mixup import Mixup
from dataset import get_dataset
import tasks
from itertools import cycle
import numpy as np
from sklearn import metrics
from tool.metrics import Evaluator, Evaluator_BCSS
from tqdm import tqdm
import logging


class Tester(object):
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset_name = opts.dataset
        classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        # print(classes)
        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            old_classes = tot_classes - new_classes
        else:
            old_classes = 0

        path_work = 'work/test/'
        if os.path.exists(path_work) is False:
            os.mkdir(path_work)

        random_seed = 42
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # np.random.seed(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        dataset_size = [224, 224]
        dataset_test = Dataset_test(dataset_size, self.device, dataset_name)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

        if opts.phase == 0:
            self.model = Swin_MIL(opts, classes=tot_classes, ema=True).to(self.device)
            # t_model = Swin_MIL(opts, classes=tot_classes, ema=True).to(device)
        else:
            self.model = Swin_MIL(opts, classes=tot_classes, ema=False).to(self.device)
    
    def load_the_best_checkpoint(self):
        if self.opts.step_ckpt is not None:
            path = self.opts.step_ckpt
        else:
            path = f'./work/test/s{self.opts.step}_{self.opts.dataset}_p{self.opts.phase}_{self.opts.weights[0]}_{self.opts.weights[1]}_{self.opts.weights[2]}_{self.opts.weights[3]}_best_model.pth'
        step_checkpoint = torch.load(path, map_location="cpu")
        # checkpoint = torch.load('checkpoints/stage2_checkpoint_trained_on_'+self.args.dataset+'.pth')
        self.model.load_state_dict(step_checkpoint, strict=True) 

    def test(self):
        self.load_the_best_checkpoint()
        self.model.eval()
        nclass = 4
        evaluator = Evaluator(nclass)
        evaluator.reset()
        loop = tqdm(self.dataloader_test, desc='\r', total=len(self.dataloader_test))

        with torch.no_grad():
            for index, sample in enumerate(loop):
                image, label = sample

                preds = self.model(image.to(self.device))
                pred = F.interpolate(preds[-1], size=(label.shape[-1], label.shape[-1]), mode='bilinear', align_corners=False)
                pred = torch.argmax(pred, dim=1).cpu().numpy()

                target = label.squeeze(1).cpu().numpy()
                ### LUAD-HistoSeg and BCSS-WSSS
                ## cls 4 is exclude
                pred[target==4]=4


                evaluator.add_batch(target, pred)

            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            ious = evaluator.Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

            print('Test:')
            # print('epoch %d loss:%.4f' % (epochs, average_loss))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('IoUs: ', ious)


            # loginfo
            weights_str = "_".join(map(str, self.opts.weights))  # Create a string from the weights list
            logging.info(f'Test Results with Weights {weights_str}:')
            logging.info(f"     Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}")
            logging.info(f'     IoUs: {ious}')






def main(opts):

    tester = Tester(opts)
    tester.test()

    # for handler in logging.getLogger().handlers:
    #     handler.flush()
    #     handler.close()

    


if __name__ == "__main__":

    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    mode = 'w' if opts.first_run else 'a'
    # print(mode)
    # print(os.path.join(opts.log_dir, f'testset_with_diff_weights.log'))
    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opts.log_dir, f'experiment.log'),
        level=logging.INFO,
        format='%(message)s',#'%(message)s',
        filemode=mode
        )

    main(opts)

# def test(path_work, model, dataloader, device, weight=None):
#     if weight is None:
#         path_model = path_work + 'best_model.pth'
#     elif weight == 'final':
#         path_model = path_work + 'final_model.pth'
#     else:
#         path_model = path_work + weight
#     model.load_state_dict(torch.load(path_model))
#     model.eval()
#     plt.ion()

#     step = 0
#     total_f = 0
#     total_hd = 0
#     total_time = 0
#     num = 0
#     split = 80
#     num_1 = 0
#     num_0 = 0
#     total_f_1 = 0
#     total_f_0 = 0
#     f1 = 0
#     loss = 0
#     max = 0
#     mask_gm = 0
#     gm = 0

#     with torch.no_grad():
#         for image, label, image_show in dataloader: #
#             # time_start = time.time()
#             step += 1

#             preds = model(image.to(device))

#             pred = ((preds[3] >= 0.5) + 0).squeeze(0).squeeze(0).to('cpu').numpy()
#             label = label.squeeze(0).squeeze(0).int().to("cpu").numpy()
#             image_show = image_show.squeeze(0).int().to("cpu").numpy()

#             if step <= split:
#                 num_1 += 1
#                 f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
#                 total_f_1 += f1
#                 average_f_1 = total_f_1 / num_1
#                 hausdorff_distance = hausdorff(pred, label)
#                 total_hd += hausdorff_distance
#                 average_hd = total_hd / num
#             else:
#                 num_0 += 1
#                 f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)
#                 total_f_0 += f1
#                 average_f_0 = total_f_0 / num_0
#         print('F1 Pos = %.3f' % average_f_1)
#         print("average HD = %.3f" % average_hd)
#         print('F1 Neg = %.3f' % average_f_0)

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(image_show)
            # plt.title("%dth F = %.3f" % (step, f1))
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(1, 3, 2)
            # plt.imshow(label, cmap='YlOrRd')
            # plt.title("Ground truth")
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(1, 3, 3)
            # plt.imshow(pred, cmap='YlOrRd')
            # plt.title("Prediction")
            # plt.xticks([])
            # plt.yticks([])
            # plt.pause(1)
            # plt.savefig("output/unet_idt/com/%04d.jpg" % step)
            # plt.show()
            # plt.imsave('output/unet_idt/vis/%04d.jpg' % step, pred, cmap='YlOrRd')

        #     if num == 1:
        #         continue
        #
        #     time_end = time.time()
        #
        #     running_time = time_end - time_start
        #     total_time += running_time
        #     average_running_time = total_time / (num - 1)
        #     print('Running Time: ', running_time)
        # print('Average Time: ', average_running_time)
