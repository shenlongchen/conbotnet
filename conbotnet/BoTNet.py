# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/15
@Auth ： shenlongchen
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conbotnet.data_utils import ACIDS
from conbotnet.modules import *

__all__ = ['BoTNet', 'LinearPredictor', 'SupConBoTNet', 'LinConBoTNet']


class EmbeddingLayer(nn.Module):
    """
        embedding layer for peptide and mhc
    """

    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        return self.peptide_emb(peptide_x.long()), self.mhc_emb(mhc_x.long()), masks

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)

        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width = x.size()  # 256, 128, 16
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)  # 256, 4, 32, 16
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)  # 256, 4, 32, 16
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)  # 256, 4, 32, 16

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 128, 4, 16, 16

        content_position = (self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)  # 1, 4, 16, 32
        content_position = torch.matmul(content_position, q)  # 128, 4, 16, 16

        energy = content_content + content_position  # ,4, 16, 16
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 128, 4, 64, 7
        out = out.view(n_batch, C, width)  # 128, 256, 7

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        if not mhsa:
            self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool1d(2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BoTNet(EmbeddingLayer):
    def __init__(self, *, conv_num, conv_size, conv_off, heads=4, dropout=0.5, pooling=True, **kwargs):
        super(BoTNet, self).__init__(**kwargs)
        self.conv_onehot = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_onehot_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)

        self.pooling = pooling

        self.conv_mhc = nn.Conv1d(21, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv_pep = nn.Conv1d(21, 64, kernel_size=3, padding=1)

        self.in_planes = 128
        self.resolution = 16
        block = Bottleneck
        num_blocks = [2, 3, 2]
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1, heads=heads, mhsa=True)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, peptide_x, mhc_x, peptide_one_hot, mhc_one_hot, pooling=None, **kwargs):
        peptide_x, mhc_x, masks = super(BoTNet, self).forward(peptide_x, mhc_x)
        peptide_one_hot = peptide_one_hot.transpose(1, 2)
        mhc_one_hot = mhc_one_hot.transpose(1, 2)

        peptide_one_hot = self.conv_pep(peptide_one_hot)
        mhc_one_hot = self.conv_mhc(mhc_one_hot)

        peptide_one_hot = peptide_one_hot.transpose(1, 2)
        mhc_one_hot = mhc_one_hot.transpose(1, 2)

        conv_one_hot_out = torch.cat(
            [conv_bn(F.relu(conv(peptide_one_hot[:, off: peptide_one_hot.shape[1] - off], mhc_one_hot)))
             for conv, conv_bn, off in zip(self.conv_onehot, self.conv_onehot_bn, self.conv_off)], dim=1)
        conv_one_hot_out = self.dropout(conv_one_hot_out)

        conv_embeded_out = torch.cat([conv_bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
                                      for conv, conv_bn, off in zip(self.conv, self.conv_bn, self.conv_off)], dim=1)
        conv_embeded_out = self.dropout(conv_embeded_out)

        conv_out = torch.cat([conv_one_hot_out, conv_embeded_out], dim=1)
        # conv_out = conv_embeded_out

        masks = masks[:, -conv_out.shape[2]:]

        botnet_input = conv_out
        out = self.layer1(botnet_input)
        if pooling or self.pooling:

            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
        else:
            # out = torch.sigmoid(torch.mean(out[:, 64:, :], dim=1)).masked_fill(~masks, -np.inf)
            out = torch.sigmoid(torch.mean(out[:, :, :], dim=1)).masked_fill(~masks, -np.inf)
        return out

    def reset_parameters(self):
        super(BoTNet, self).reset_parameters()
        self.conv_mhc.reset_parameters()
        self.conv_pep.reset_parameters()
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)


class SupConBoTNet(nn.Module):
    def __init__(self, *, conv_num, conv_size, conv_off, heads=4, dropout=0.5, pooling=True, **kwargs):
        super(SupConBoTNet, self).__init__()
        self.encoder = BoTNet(conv_num=conv_num, conv_size=conv_size, conv_off=conv_off, heads=heads, dropout=dropout,
                              pooling=pooling, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16)
        )

    def forward(self, peptide_x, mhc_x, peptide_one_hot, mhc_one_hot, pooling=None, **kwargs):
        feature = self.encoder(peptide_x, mhc_x, peptide_one_hot, mhc_one_hot, pooling=pooling, **kwargs)
        feature = F.normalize(self.head(feature), dim=1)
        return feature


class LinearPredictor(nn.Module):
    def __init__(self, input_size):
        super(LinearPredictor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x).flatten())

    def reset_parameters(self):
        self.fc.reset_parameters()


class LinConBoTNet(nn.Module):
    def __init__(self, **kwargs):
        super(LinConBoTNet, self).__init__()
        self.network = SupConBoTNet(**kwargs).cuda()
        self.classifier = LinearPredictor(256).cuda()

    def forward(self, inputs, **kwargs):
        features = self.network.encoder(*(x.cuda() for x in inputs), **kwargs)
        output = self.classifier(features)
        return output


class BinConBoTNet(nn.Module):
    def __init__(self, **kwargs):
        super(BinConBoTNet, self).__init__()
        self.network = SupConBoTNet(**kwargs).cuda()
        self.classifier = LinearPredictor(256).cuda()

    def forward(self, inputs, **kwargs):
        features = self.network.encoder(*(x.cuda() for x in inputs), **kwargs)
        return features
