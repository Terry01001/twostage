import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
from .mix_transformer import *
from module.modules import ASPP
from .segformer_head import SegFormerHead
from .unet_decoder import *
import numpy as np

class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(512, num_classes, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(128, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(320, 128, 1, bias=False)
        
        self.f9_1 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        self.f9_2 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9_1.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.f9_2.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9_1, self.f9_2, self.fc8]
        
    def get_norm_cam_d(self, cam):
        """normalize the activation vectors of each pixel by supressing foreground non-maximum activations to zeros"""
        n, c, h, w = cam.size() # [2, 4, 32, 32]
        with torch.no_grad():
            cam_d = cam.detach()
            cam_d_min = torch.min(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5 # [2, 4, 1, 1] each channel has a max value (for each image in batch)
            cam_d_norm = (cam - cam_d_min) / (cam_d_max - cam_d_min) # [2, 4, 32, 32]
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0] # background channel is 0, which is calculated by 1 - max(other channels)
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0] # [2, 1, 32, 32], max value of each channel (except background channel)
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0 # set the non-max value to 0
            
        return cam_d_norm

    def forward(self, x, x1, x2, deep3, _4): # x2: [32,128,28,28] deep3: [32,320,14,14] _4: [32,512,14,14]
        N, C, H, W = x.size()  # [32, 3, 224, 224]
        

        cam = self.fc8(self.dropout7(_4)) 
        n, c, h, w = cam.size() # [32,4,14,14]

        cam_d_norm = self.get_norm_cam_d(cam)
        
        
        # ----> Get Concated Feature
        x2 = F.interpolate(x2, (h,w), mode='bilinear', align_corners=True)
        f8_3 = F.relu(self.f8_3(x2), inplace=True) 
        f8_4 = F.relu(self.f8_4(deep3), inplace=True)
        
        x_s = F.interpolate(x, (h, w), mode='bilinear',align_corners=True) 
        f = torch.cat([x_s, f8_3, f8_4], dim=1) # [32, 192+3, 14, 14]
        n, c, h, w = f.size() # [32, 192+3, 14, 14]
        
        # ----> Attention
        q = self.f9_1(f).view(n, -1, h*w) 
        # q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-5)
        k = self.f9_2(f).view(n, -1, h*w) 
        # k = k / (torch.norm(k, dim=1, keepdim=True) + 1e-5)
        A = torch.matmul(q.transpose(1, 2), k) 
        A = F.softmax(A, dim=1) # normalize over column # [32,196,196]
        assert not torch.isnan(A).any(), A
        
        # pmask_refine = self.RFM(pmask_d_norm, A, h, w)
        # pmask_rv = F.interpolate(pmask_refine, (H, W), mode='bilinear', align_corners=True)
        
        # pcam_refine = self.RFM(pcam_d_norm, A, h, w)
        # pcam_rv = F.interpolate(pcam_refine, (H, W), mode='bilinear', align_corners=True)
        
        cam_refine = self.RFM(cam_d_norm, A, h, w) # [32,4,14,14]

        # cam_rv = F.interpolate(cam_refine, (H, W), mode='bilinear', align_corners=True) 
        
        # cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True) 
        
        return cam_refine

    def RFM(self, cam, A, h, w): 
        n = A.size()[0]
        
        #cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h*w) 
        cam = cam.view(n, -1, h*w)
        cam_rv = torch.matmul(cam, A).view(n, -1, h, w)
        
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



class Swin_MIL(nn.Module):
    def __init__(self, opts, classes=0, ema=False):
        super(Swin_MIL, self).__init__()
        self.ema = ema
        self.num_class = classes
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
                #nn.Conv2d(512, classes, 1), 
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            # new add
            self.refine_module = Net(num_classes=classes)
            



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
                #nn.Conv2d(512, classes, 1),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
                nn.Sigmoid()
            )
            self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, 
                                         embedding_dim=self.embedding_dim, num_classes=classes)
            
            self.refine_module = Net(num_classes=classes)
        

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
            _4_refine = self.refine_module(x,x1,x2,deep3,_4)

            x2 = self.decoder2(x2)
            x3 = self.decoder3(deep3)
            x4 = self.decoder4(_4_refine) # change

            # x = self.w[0] * x4 + self.w[1] * x3
            
            x = self.w[0] * x2 + self.w[1] * x3 + self.w[2] * x4
            # x = (x2+x3+x4)/3
            # print(x.shape)
            # x = 0.6*x3 + 0.4*x2
            # x = x3
            


            # new add
            return x2, x3, x4, x

        else:
            x1, x2, deep3, _4 = _x
            _4_refine = self.refine_module(x, x1, x2, deep3, _4)
            x2 = self.decoder2(x2)
            x3 = self.decoder3(deep3)
            x4 = self.decoder4(_4_refine) # change
            cls4 = nn.functional.adaptive_max_pool2d(_4_refine,(1,1))
            cls4 = self.classifier(cls4)
            cls4 = cls4.view(-1, 4) # luad & bcss
            seg = self.decoder(_x)
            
            x = self.w[0] * x2 + self.w[1] * x3 + self.w[2] * x4
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
        # x = self.w[0] * x2 + self.w[1] * x3 + self.w[2] * x4
        # x = x3

        # x = self.up1(_4, deep3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)
        # # x = self.up4(x, x1)
        # logits = self.outc(x)

        # return x1, x2, x3, x, _attns, deep3, seg
        return cls4, x2, x3, x4_refine, x, seg
