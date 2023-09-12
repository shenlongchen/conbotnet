# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/15
@Auth ： shenlongchen
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from logzero import logger
from typing import Optional, Mapping, Tuple

from conbotnet.evaluation import get_auc, get_pcc, get_group_metrics
from conbotnet.losses import SupConLoss
from conbotnet.evaluation import CUTOFF
__all__ = ['ModelPretrain']


class ModelPretrain(object):
    """

    """

    def __init__(self, network, model_path, **kwargs):
        self.model = self.network = network(**kwargs).cuda()
        self.model_path = Path(model_path)
        self.criterion = SupConLoss(temperature=0.01)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

    def get_scores(self, inputs, **kwargs):
        return self.model(*(x.cuda() for x in inputs), **kwargs)

    def loss_and_backward(self, features, targets):
        features = features.unsqueeze(1)
        loss = self.criterion(features, targets.cuda())
        loss.backward()
        return loss

    def train_step(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor,
                   **kwargs):
        self.optimizer.zero_grad()
        self.model.train()
        loss = self.loss_and_backward(self.get_scores(inputs, **kwargs), targets)
        self.optimizer.step(closure=None)
        return loss.item()

    @torch.no_grad()
    def predict_step(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], **kwargs):
        self.model.eval()
        return self.get_scores(inputs, **kwargs)

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = (),
              num_epochs=20, verbose=True, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.training_state['best'] = 1000
        self.training_state['best_epoch'] = 0
        for epoch_idx in range(num_epochs):
            train_loss = 0.0
            for inputs, targets in train_loader:
                train_loss += self.train_step(inputs, targets, **kwargs) * len(targets)
            train_loss /= len(train_loader.dataset)
            valid_loss = self.valid(valid_loader, verbose, epoch_idx, train_loss, **kwargs)
            logger.info(f'Epoch: {epoch_idx} '
                        f'-- Loss: {train_loss:.5f} '
                        f'-- Valid: {valid_loss:.5f} ')
            # if epoch_idx % 5 == 0:
            #     save_file = self.model_path.with_stem(f'{self.model_path.stem}-epoch-{epoch_idx}')
            #     torch.save(self.model.state_dict(), save_file)
        logger.info(f'Best Epoch: {self.training_state["best_epoch"]} ')


    def get_loss(self, features, targets):
        features = features.unsqueeze(1)
        loss = self.criterion(features, targets.cuda())
        return loss

    def valid(self, valid_loader, verbose, epoch_idx, train_loss, **kwargs):
        scores, targets = self.predict(valid_loader, valid=True, **kwargs)
        valid_loss = self.get_loss(scores, targets)

        if valid_loss < self.training_state['best']:
            self.save_model()
            self.training_state['best'] = valid_loss
            self.training_state['best_epoch'] = epoch_idx
        return valid_loss

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        features = []
        sup_targets = []
        for data_x, targets in data_loader:
            features.append(self.predict_step(data_x, **kwargs))
            sup_targets.append(targets)
        features = torch.cat(features, dim=0)
        sup_targets = torch.cat(sup_targets, dim=0)
        return features, sup_targets

    def save_model(self):
        save_file = self.model_path.with_stem(f'{self.model_path.stem}-epoch-best')
        torch.save(self.model.state_dict(), save_file)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
