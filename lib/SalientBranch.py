import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.optim import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbones.SwinTransformer import SwinB


class SalientBranch(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384], **kwargs):
        super(SalientBranch, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size

        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=self.base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=self.base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=self.base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=self.base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=self.base_size, stage=4)

        self.decoder = PAA_d(self.depth * 3, depth=self.depth, base_size=base_size, stage=2)

        self.attention0 = SICA(self.depth    , depth=self.depth, base_size=self.base_size, stage=0, lmap_in=True)
        self.attention1 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=1, lmap_in=True)
        self.attention2 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=2              )

        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x, y, reduction='mean')
        self.pc_loss_fn = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')

        self.transition0 = Transition(17)
        self.transition1 = Transition(9)
        self.transition2 = Transition(5)

    def forward_encode(self, x):
        B, _, H, W = x.shape
    
        x1, x2, x3, x4, x5 = self.backbone(x)
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder([x3, x4, x5]) #8
        
        return f3


def SalientBranch_Res2Net50(depth, pretrained, base_size, **kwargs):
    return SalientBranch(res2net50_v1b_26w_4s(pretrained=pretrained), [64, 256, 512, 1024, 2048], depth, base_size, **kwargs)


def SalientBranch_SwinB(depth, pretrained, base_size, **kwargs):
    return SalientBranch(SwinB(pretrained=pretrained), [128, 128, 256, 512, 1024], depth, base_size, **kwargs)
