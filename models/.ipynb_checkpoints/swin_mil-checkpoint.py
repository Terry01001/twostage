import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
from .mix_transformer import *
from module.modules import ASPP
from .segformer_head import SegFormerHead
from .unet_decoder import *
import numpy as np


class Swin_MIL(nn.Module):
    def __init__(self, opts, classes=0):
        super(Swin_MIL, self).__init__()
        if opts.backbone == 'swin_t':
            self.backbone = SwinTransformer(depths=[2, 2, 6, 2], out_indices=(0, 1, 2))
            self.backbone_name = 'swin_t'
            # self.backbone = mit_b4(stride=[4, 2, 2, 1])
            # self.asppf = ASPP(4)
            # self.conv1 = nn.Conv2d(12, 4, 1)
            # self.decoder1 = nn.Sequential(
            #     nn.Conv2d(96, 4, 1),
            #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     nn.Sigmoid()
            # )
            self.decoder1 = nn.Sequential(
                nn.Conv2d(96, classes, 1),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder2 = nn.Sequential(
                nn.Conv2d(192, classes, 1),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder3 = nn.Sequential(
                nn.Conv2d(384, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
        else:
            self.stride = opts.output_stride
            self.feature_strides = [4, 8, 16, 32]
            self.embedding_dim = 256
            self.backbone = mit_b4(stride=self.stride)
            self.in_channels = self.backbone.embed_dims
            ######## u-net decoder ###########
            # bilinear = False
            # factor = 2 if bilinear else 1
            # self.up1 = (Up(512, 320 // factor, bilinear))
            # self.up2 = (Up(320, 128 // factor, bilinear))
            # self.up3 = (Up(128, 64 // factor, bilinear))
            # self.outc = (OutConv(64, classes))
            #####################################

            self.backbone_name = 'mit_b4'
            self.decoder1 = nn.Sequential(
                nn.Conv2d(64, classes, 1),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder2 = nn.Sequential(
                nn.Conv2d(128, classes, 1),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder3 = nn.Sequential(
                nn.Conv2d(320, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
             
            # self.ms_decoder = nn.Sequential(
            #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(),
            #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(),
            #     nn.Conv2d(256, classes, kernel_size=1, stride=1)
            # )
            self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, 
                                         embedding_dim=self.embedding_dim, num_classes=classes)

        # self.attn_proj = nn.Conv2d(in_channels=36, out_channels=1, kernel_size=1, bias=True)
        # nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        # self.classifier = nn.Conv2d(in_channels=672, out_channels=4, kernel_size=1, bias=False)

        # self.pooling = F.adaptive_max_pool2d
        

        self.w = [0.3, 0.4, 0.3]
        # self.w = [0.3, 0.3, 0.4]

    def pretrain(self, opts, model, device):
        for w in model.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.xavier_uniform_(w.weight)

        
        # if opts.step > 0:
        #     # pretrained_dict = torch.load(f'work/test/s{opts.step - 1}_best_model.pth')['model']
        #     pretrained_dict = torch.load(f'work/test/s{opts.step - 1}_best_model.pth')
        # else:
        if opts.backbone == 'swin_t':
            model.backbone.init_weights()
            model_dict = model.state_dict()
            pretrained_dict = torch.load('pretrained_models/swin_tiny_patch4_window7_224.pth')['model'] #origin
        else:
            model_dict = model.state_dict()
            pretrained_path = f'pretrained_models/{opts.backbone}.pth'
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        new_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if 'backbone.' + k in model_dict.keys()}
        model_dict.update(new_dict)
        # model_dict.pop('head.weight')
        # model_dict.pop('head.bias')
        model.load_state_dict(model_dict)

        return model

    def forward(self, x, cam_only=False):
        # x1, x2, deep3, x4 = self.backbone(x)
        _x, _attns = self.backbone(x)
        if self.backbone_name == 'swin_t':
            x1, x2, deep3 = _x
        else:
            x1, x2, deep3, _4 = _x
            seg = self.decoder(_x)
            # print(_4.shape)
            # x = self.up1(_4, deep3)
            # x = self.up2(x, x2)
            # x = self.up3(x, x1)
            # seg = self.outc(x)

        # x2_f = F.interpolate(x2, size=(x1.shape[2],x1.shape[3]), mode='bilinear', align_corners=False)
        # x3_f = F.interpolate(deep3, size=(x1.shape[2],x1.shape[3]), mode='bilinear', align_corners=False)
        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(deep3)

        x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3

        # x = self.up1(_4, deep3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # # x = self.up4(x, x1)
        # logits = self.outc(x)

        # return x1, x2, x3, x, _attns, deep3, seg
        return x1, x2, x3, x, _attns, deep3, seg
