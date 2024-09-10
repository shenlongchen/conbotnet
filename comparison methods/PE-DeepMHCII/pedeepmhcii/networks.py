#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pedeepmhcii.data_utils import ACIDS
from pedeepmhcii.modules import *
from pedeepmhcii.init import truncated_normal_


class Network(nn.Module):
    """

    """
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(Network, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

        self.peptide_pos_emb = nn.Parameter(self.positional_encoding(21, emb_size),
                                            requires_grad=False)
        self.mhc_pos_emb = nn.Parameter(self.positional_encoding(mhc_len, emb_size), requires_grad=False)

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        # masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        # return self.peptide_emb(peptide_x), self.mhc_emb(mhc_x), masks

        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx


        peptide_emb = self.peptide_emb(peptide_x)
        mhc_emb = self.mhc_emb(mhc_x)

        position_encoding = self.peptide_pos_emb.unsqueeze(0).expand(peptide_x.size(0), -1, -1)
        mask_expanded = masks.unsqueeze(-1).expand_as(position_encoding)

        pos_encodings_masked = position_encoding * mask_expanded.float()
        peptide_emb[:, self.peptide_pad:self.peptide_pad + 21, :] += pos_encodings_masked

        mhc_emb = mhc_emb + self.mhc_pos_emb.unsqueeze(0).expand(mhc_x.size(0), -1, -1)

        return peptide_emb, mhc_emb, masks

    def positional_encoding(self, length, emb_size):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(np.log(10000.0) / emb_size))
        pos_emb = torch.zeros(length, emb_size)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb
    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class PEDeepMHCII(Network):
    """

    """
    def __init__(self, *, conv_num, conv_size, conv_off, linear_size, dropout=0.5, pooling=True, **kwargs):
        super(PEDeepMHCII, self).__init__(**kwargs)
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)
        linear_size = [sum(conv_num)] + linear_size
        self.linear = nn.ModuleList([nn.Conv1d(in_s, out_s, 1)
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])
        self.output = nn.Conv1d(linear_size[-1], 1, 1)
        self.pooling = pooling
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, pooling=None, **kwargs):
        peptide_x, mhc_x, masks = super(PEDeepMHCII, self).forward(peptide_x, mhc_x)
        conv_out = torch.cat([conv_bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
                              for conv, conv_bn, off in zip(self.conv, self.conv_bn, self.conv_off)], dim=1)
        conv_out = self.dropout(conv_out)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            conv_out = linear_bn(F.relu(linear(conv_out)))
        conv_out = self.dropout(conv_out)
        masks = masks[:, None, -conv_out.shape[2]:]
        if pooling or self.pooling:
            pool_out, _ = conv_out.masked_fill(~masks, -np.inf).max(dim=2, keepdim=True)
            return torch.sigmoid(self.output(pool_out).flatten())
        else:
            return torch.sigmoid(self.output(conv_out)).masked_fill(~masks, -np.inf).squeeze(1)

    def reset_parameters(self):
        super(PEDeepMHCII, self).reset_parameters()
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        truncated_normal_(self.output.weight, std=0.1)
        nn.init.zeros_(self.output.bias)
