# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/15
@Auth ： shenlongchen
"""

import os
import numpy as np
import torch
from conbotnet.aminoacids import to_onehot
from torch.utils.data.dataset import Dataset
from conbotnet.data_utils import ACIDS
import math

__all__ = ['MHCIIDataset', 'EOMHCIIDataset']


class MHCIIDataset(Dataset):
    """

    """

    def __init__(self, data_list, peptide_len=20, peptide_pad=3, mhc_len=34, padding_idx=0):
        self.mhc_names, self.peptide_x, self.mhc_x, self.targets = [], [], [], []
        for mhc_name, peptide_seq, mhc_seq, score in data_list:
            self.mhc_names.append(mhc_name)
            peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq][:peptide_len]
            self.peptide_x.append([padding_idx] * peptide_pad +
                                  peptide_x + [padding_idx] * (peptide_len - len(peptide_x)) +
                                  [padding_idx] * peptide_pad)
            assert len(self.peptide_x[-1]) == peptide_len + peptide_pad * 2
            self.mhc_x.append([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])
            assert len(self.mhc_x[-1]) == mhc_len
            self.targets.append(score)
        self.peptide_x, self.mhc_x = np.asarray(self.peptide_x), np.asarray(self.mhc_x)
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __getitem__(self, item):
        return (self.peptide_x[item], self.mhc_x[item]), self.targets[item]

    def __len__(self):
        return len(self.mhc_names)


class EOMHCIIDataset(Dataset):
    def __init__(self, data_list, peptide_len=20, peptide_pad=3, mhc_len=34, padding_idx=0):
        self.mhc_names, self.peptide_x, self.mhc_x, self.targets = [], [], [], []
        self.peptide_seqs, self.mhc_seqs = [], []
        self.peptide_len = peptide_len
        self.max_len_peptide = peptide_len + peptide_pad * 2
        self.peptide_pad = peptide_pad
        self.max_len_mhc = mhc_len
        for mhc_name, peptide_seq, mhc_seq, score in data_list:
            self.mhc_names.append(mhc_name)
            peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq][:peptide_len]
            self.peptide_x.append([padding_idx] * peptide_pad +
                                  peptide_x + [padding_idx] * (peptide_len - len(peptide_x)) +
                                  [padding_idx] * peptide_pad)
            assert len(self.peptide_x[-1]) == peptide_len + peptide_pad * 2
            self.mhc_x.append([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])
            assert len(self.mhc_x[-1]) == mhc_len
            self.peptide_seqs.append(peptide_seq)
            self.mhc_seqs.append(mhc_seq)
            self.targets.append(score)
        self.peptide_x, self.mhc_x = np.asarray(self.peptide_x), np.asarray(self.mhc_x)
        self.peptide_seqs, self.mhc_seqs = np.asarray(self.peptide_seqs), np.asarray(self.mhc_seqs)
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __getitem__(self, item):
        peptide_one_hot = to_onehot(self.peptide_seqs[item][: self.peptide_len], max_len=self.max_len_peptide,
                                    start=self.peptide_pad)
        mhc_one_hot = to_onehot(self.mhc_seqs[item], max_len=self.max_len_mhc)
        return (self.peptide_x[item], self.mhc_x[item], peptide_one_hot, mhc_one_hot), self.targets[item]

    def __len__(self):
        return len(self.mhc_names)
