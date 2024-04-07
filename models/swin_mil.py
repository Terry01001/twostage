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
    def __init__(self, opts, classes=0, ema=False):
        super(Swin_MIL, self).__init__()
        self.ema = ema
        if opts.backbone == 'swin_t':
            self.backbone = SwinTransformer(depths=[2, 2, 6, 2], out_indices=(0, 1, 2))
            self.backbone_name = 'swin_t'
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
        elif opts.backbone == 'mit_b4' and self.ema:
            self.stride = opts.output_stride
            self.backbone = mit_b4(stride=self.stride)
            self.in_channels = self.backbone.embed_dims
            self.backbone_name = 'mit_b4'
            # self.decoder1 = nn.Sequential(
            #     nn.Conv2d(64, classes, 1),
            #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     nn.Sigmoid()
            # )
            self.decoder2 = nn.Sequential(
                nn.Conv2d(128, classes, 1),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )

            # self.decoder2 = nn.Sequential(
            #     # nn.Conv2d(self.in_channels[1], self.in_channels[1] // 2, 1),
            #     nn.ConvTranspose2d(self.in_channels[1], self.in_channels[1] // 2, kernel_size=4, stride=4),
            #     nn.BatchNorm2d(self.in_channels[1] // 2),
            #     nn.ReLU(),
            #     # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     # nn.ConvTranspose2d(self.in_channels[1], self.in_channels[1] // 2, kernel_size=4, stride=4),
            #     # nn.Dropout2d(0.2),
            #     # nn.Conv2d(self.in_channels[1] // 2, classes, 1),
            #     nn.ConvTranspose2d(self.in_channels[1] // 2, self.in_channels[1] // 2, kernel_size=2, stride=2),
            #     nn.BatchNorm2d(self.in_channels[1] // 2),
            #     nn.ReLU(),
            #     nn.Conv2d(self.in_channels[1] // 2, classes, 1),
            #     # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            #     nn.Sigmoid()
            # )

            # self.decoder3 = nn.Sequential(
            #     # nn.Conv2d(self.in_channels[2], self.in_channels[2] // 2, 1),
            #     nn.ConvTranspose2d(self.in_channels[2], self.in_channels[2] // 2, kernel_size=4, stride=4),
            #     nn.BatchNorm2d(self.in_channels[2] // 2),
            #     nn.ReLU(),
            #     # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     # nn.Dropout2d(0.2),
            #     # nn.Conv2d(self.in_channels[2] // 2, classes, 1),
            #     nn.ConvTranspose2d(self.in_channels[2] // 2, self.in_channels[2] // 2, kernel_size=4, stride=4),
            #     nn.BatchNorm2d(self.in_channels[2] // 2),
            #     nn.ReLU(),
            #     nn.Conv2d(self.in_channels[2] // 2, classes, 1),
            #     # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     nn.Sigmoid()
            # )
            # self.decoder4 = nn.Sequential(
            #     # nn.Conv2d(self.in_channels[2], self.in_channels[2] // 2, 1),
            #     nn.ConvTranspose2d(self.in_channels[3], self.in_channels[3] // 2, kernel_size=4, stride=4),
            #     nn.BatchNorm2d(self.in_channels[3] // 2),
            #     nn.ReLU(),
            #     # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     # nn.Dropout2d(0.2),
            #     # nn.Conv2d(self.in_channels[2] // 2, classes, 1),
            #     nn.ConvTranspose2d(self.in_channels[3] // 2, self.in_channels[3] // 2, kernel_size=4, stride=4),
            #     nn.BatchNorm2d(self.in_channels[3] // 2),
            #     nn.ReLU(),
            #     nn.Conv2d(self.in_channels[3] // 2, classes, 1),
            #     # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            #     nn.Sigmoid()
            # )
            self.decoder3 = nn.Sequential(
                nn.Conv2d(320, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder4 = nn.Sequential(
                nn.Conv2d(512, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            # #TODO:
            # self.dropout7 = torch.nn.Dropout2d(0.5)

            # self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)

            # self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
            # self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        
            # self.f9_1 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
            # self.f9_2 = torch.nn.Conv2d(192+3, 192, 1, bias=False)

            # torch.nn.init.xavier_uniform_(self.fc8.weight)
            # torch.nn.init.kaiming_normal_(self.f8_3.weight)
            # torch.nn.init.kaiming_normal_(self.f8_4.weight)
            # torch.nn.init.xavier_uniform_(self.f9_1.weight, gain=4)
            # torch.nn.init.xavier_uniform_(self.f9_2.weight, gain=4)



        else:
            self.stride = opts.output_stride
            self.feature_strides = [4, 8, 16, 32]
            self.embedding_dim = 256
            self.backbone = mit_b4(stride=self.stride)
            self.in_channels = self.backbone.embed_dims
            self.backbone_name = 'mit_b4'
            self.decoder2 = nn.Sequential(
                nn.Conv2d(128, classes, 1),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            # self.decoder2 = nn.Sequential(
            #     nn.Conv2d(self.in_channels[1], self.in_channels[1] // 2, 1),
            #     nn.BatchNorm2d(self.in_channels[1] // 2),
            #     nn.ReLU,
            #     nn.Dropout2d(0.5),
            #     nn.Conv2d(self.in_channels[1] // 2, classes, 1),
            #     nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            #     nn.Sigmoid()
            # )
            self.decoder3 = nn.Sequential(
                nn.Conv2d(320, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder4 = nn.Sequential(
                nn.Conv2d(512, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, 
                                         embedding_dim=self.embedding_dim, num_classes=classes)
        

        self.w = [0.3, 0.4, 0.3]
        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=classes, kernel_size=1, bias=False)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")
        # self.w = [0.2, 0.2, 0.3, 0.3]

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
        # print(x.shape)
        # print(x.sadasdada())
        _x, _attns = self.backbone(x)
        # attn_cat = torch.cat(_attns[-2:], dim=1)
        # attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        # attn_pred = self.attn_proj(attn_cat)
        # print(attn_pred.shape)
        # attn_pred = torch.sigmoid(attn_pred)[:,0,...]

        # print(attn_pred.shape)
        # for at in _attns:
        #     print(at.shape)
        # print(x.asdasdasdsd())
        if self.backbone_name == 'swin_t':
            x1, x2, deep3 = _x
            x1 = self.decoder1(x1)
            x2 = self.decoder2(x2)
            x3 = self.decoder3(deep3)

            x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3
            return x1, x2, x3, x
        elif self.backbone_name == 'mit_b4' and self.ema:
            x1, x2, deep3, _4 = _x
            # x1 = self.decoder1(x1)

            x2 = self.decoder2(x2)
            x3 = self.decoder3(deep3)
            x4 = self.decoder4(_4)

            # x = self.w[0] * x4 + self.w[1] * x3
            x = self.w[0] * x2 + self.w[1] * x3 + self.w[2] * x4
            # x = (x2+x3+x4)/3
            # print(x.shape)
            # x = 0.6*x3 + 0.4*x2
            # x = x3
            return x2, x3, x4, x


            # #TODO:
            # f8_3 = F.relu(x2, inplace=True) 
            # f8_4 = F.relu(x3, inplace=True)

            # x_s = F.interpolate(x2, (h, w), mode='bilinear',align_corners=True) 
            # f = torch.cat([x_s, f8_3, f8_4], dim=1) 
            # n, c, h, w = f.size() 
            
            # # ----> Attention
            # q = self.f9_1(f).view(n, -1, h*w) # [2, 192, 32*32] 
            # # q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-5)
            # k = self.f9_2(f).view(n, -1, h*w) # [2, 192, 32*32]
            # # k = k / (torch.norm(k, dim=1, keepdim=True) + 1e-5)
            # A = torch.matmul(q.transpose(1, 2), k) # [2, 32*32, 32*32]
            # A = F.softmax(A, dim=1) # normalize over column
            # assert not torch.isnan(A).any(), A



            return x2, x3, x4, x
        else:
            x1, x2, deep3, _4 = _x
            x2 = self.decoder2(x2)
            x3 = self.decoder3(deep3)
            x4 = self.decoder4(_4)
            cls4 = nn.functional.adaptive_max_pool2d(_4,(1,1))
            cls4 = self.classifier(cls4)
            cls4 = cls4.view(-1, 4) # luad & bcss
            seg = self.decoder(_x)
            # print(_4.shape)
            # x = self.up1(_4, deep3)
            # x = self.up2(x, x2)
            # x = self.up3(x, x1)
            # seg = self.outc(x)

        if cam_only:
            cam_s4 = F.conv2d(_4, self.classifier.weight).detach()
            return cam_s4
        # x2_f = F.interpolate(x2, size=(x1.shape[2],x1.shape[3]), mode='bilinear', align_corners=False)
        # x3_f = F.interpolate(deep3, size=(x1.shape[2],x1.shape[3]), mode='bilinear', align_corners=False)
        # x1 = self.decoder1(x1)
        # x2 = self.decoder2(x2)
        # x3 = self.decoder3(deep3)

        # x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3
        # x = 0.5*x3 + 0.5*x2
        x = self.w[0] * x2 + self.w[1] * x3 + self.w[2] * x4
        # x = x3

        # x = self.up1(_4, deep3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # # x = self.up4(x, x1)
        # logits = self.outc(x)

        # return x1, x2, x3, x, _attns, deep3, seg
        return cls4, x2, x3, x4, x, seg
