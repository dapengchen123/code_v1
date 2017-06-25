from __future__ import absolute_import
import torch.nn.functional as F
import torch.nn.init as init

from torch import nn
from torch.autograd import Variable

from torchvision.models import resnet18, resnet34, resnet50, resnet101, \
    resnet152

from torch import zeros as t_zeros

LSTM_hidden = 128
LSTM_layer = 2


def init_states(hidden_sz, batch_sz, layer=1):
    h_0 = Variable(t_zeros(layer, batch_sz, hidden_sz))
    c_0 = Variable(t_zeros(layer, batch_sz, hidden_sz))
    return (h_0.cuda(), c_0.cuda())


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
        self.has_embedding = num_features > 0


        ### Append new upper layers
        out_planes = self.base.fc.in_features

        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal(self.feat.weight, mode='fan_out')
            init.constant(self.feat.bias, 0)
            init.constant(self.feat_bn.weight, 1)
            init.constant(self.feat_bn.bias, 0)
        else:
            self.num_features = out_planes

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        ### Append LSTM layers
        self.lstm = nn.LSTM(self.num_features, LSTM_hidden, num_layers=LSTM_layer, batch_first=True)


    def forward(self, imgs, motions):

        img_size = imgs.size()
        motion_size = motions.size()

        batch_sz = img_size[0]
        seq_len = img_size[1]


        imgs = imgs.view(img_size[0] * img_size[1], img_size[2], img_size[3], img_size[4])
        motions = motions.view(motion_size[0] * motion_size[1], motion_size[2], motion_size[3], motion_size[4])
        motions = motions[:, 1:3]
        for name, module in self.base._modules.items():
            if name == 'conv1':
                x = module(imgs)+self.conv0(motions)
                continue

            if name == 'avgpool':
                break

            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.norm:
            x = x / x.norm(2, 1).expand_as(x)
        elif self.has_embedding:
            x = F.relu(x)

        if self.dropout > 0:
            x = self.drop(x)

        ######  LSTM  part #########
        x = x.view(batch_sz, seq_len, -1)

        hidden0 = init_states(LSTM_hidden, batch_sz, LSTM_layer)
        output, _ = self.lstm (x, hidden0)
        return  output[:, -1]














