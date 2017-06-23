from __future__ import absolute_import
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, \
    resnet152


class ResNetLSTM_btfu(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, norm=False, dropout=0):
        super(ResNetLSTM_btfu, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNetLSTM_btfu.__factory:
            raise KeyError("Unsupported depth:". depth)


        ### At the bottom of CNN network
        conv0 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal(conv0.weight, mode='fan_out')
        self.conv0 = conv0
        self.base = ResNetLSTM_btfu.__factory[depth](pretrained=pretrained)

        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout


        ### Append new upper layers
        out_planes = self.base.fc.in_features
        self.feat = nn.Linear(out_planes, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        ### Append LSTM layers
        









