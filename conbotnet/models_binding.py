# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/15
@Auth ： shenlongchen
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from torch.utils.data import DataLoader

__all__ = ['ModelBinding']


class ModelBinding(object):
    """

    """

    def __init__(self, network, model_path, **kwargs):
        self.model = self.network = network(**kwargs).cuda()

        self.loss_fn, self.model_path = nn.MSELoss(), Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

        trained_model = model_path.parent.parent.joinpath('fine_tuning').joinpath(self.model_path.stem + '.pt')
        self.model.load_state_dict(torch.load(trained_model))
        logger.info(f'Loading fine-tuning model from {trained_model}')

    def get_scores(self, inputs, **kwargs):
        # features = self.model.encoder(*(x.cuda() for x in inputs), **kwargs)
        features = self.model(inputs, **kwargs)
        # output = self.classifier(features)
        return features

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs).cpu()

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        return np.concatenate([self.predict_step(data_x, **kwargs) for data_x, _ in data_loader], axis=0)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
