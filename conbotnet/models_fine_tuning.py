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
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping, Tuple
from conbotnet.BoTNet import LinearPredictor

from conbotnet.evaluation import get_auc, get_pcc, get_group_metrics
# from data_processing.utils import get_auc, get_pcc, get_group_metrics
__all__ = ['ModelFineTuning']


class ModelFineTuning(object):
    """

    """

    def __init__(self, network, model_path, **kwargs):
        self.model = self.network = network(**kwargs).cuda()

        self.loss_fn, self.model_path = nn.MSELoss(), Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.training_state = {}

        pretrain_model = model_path.parent.parent.joinpath('pre_models').joinpath(self.model_path.stem+'-epoch-best.pt')
        if pretrain_model.exists():
            self.model.network.load_state_dict(torch.load(pretrain_model))
            logger.info(f'Loading pretrain model from {pretrain_model}')
        else:
            logger.info(f'No pretrain model found in {pretrain_model}')


    def get_scores(self, inputs, **kwargs):
        output = self.model(inputs, **kwargs)
        return output

    def loss_and_backward(self, scores, targets):
        loss = self.loss_fn(scores, targets.cuda())
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
        return self.get_scores(inputs, **kwargs).cpu()

    def get_optimizer(self, optimizer_cls='Adadelta', weight_decay=1e-3, **kwargs):
        if isinstance(optimizer_cls, str):
            optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer = optimizer_cls(self.model.parameters(), weight_decay=weight_decay, **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader = None,
              data_group_name=None, cv_id=None, cv_=None, opt_params: Optional[Mapping] = (), num_epochs=20, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.training_state['best'] = 0.0
        self.training_state['best_epoch'] = 0
        for epoch_idx in range(num_epochs):
            train_loss = 0.0

            for inputs, targets in train_loader:
                train_loss += self.train_step(inputs, targets, **kwargs) * len(targets)
            train_loss /= len(train_loader.dataset)
            auc_valid, pcc_valid = self.valid(valid_loader, epoch_idx)

            if test_loader is None:
                logger.info(f'Epoch: {epoch_idx} '
                            f'-- Loss: {train_loss:.5f} '
                            f'-- Valid: AUC: {auc_valid:.5f} PCC: {pcc_valid:.3f} ')
                continue
            if cv_id is not None:
                mhc_names = np.asarray(data_group_name)[cv_id == cv_]
            else:
                mhc_names = np.asarray(data_group_name)
            auc_all, auc_group, pcc_group, srcc_group = self.test(test_loader, mhc_names)

            logger.info(f'Epoch: {epoch_idx} '
                        f'-- Loss: {train_loss:.5f} '
                        f'-- Valid: AUC: {auc_valid:.5f} PCC: {pcc_valid:.3f} '
                        f'-- Test: All AUC: {auc_all:.5f} - Group AUC: {auc_group:.5f} - PCC: '
                        f'{pcc_group:.3f} SRCC: {srcc_group:.3f}')

            if epoch_idx % 5 == 0:
                save_file = self.model_path.with_stem(f'{self.model_path.stem}-epoch-{epoch_idx}')
                torch.save(self.model.state_dict(), save_file)
        logger.info(f'Best Epoch: {self.training_state["best_epoch"]} ')

    def valid(self, valid_loader, epoch_idx, **kwargs):
        scores, targets = self.predict(valid_loader, valid=True, **kwargs), valid_loader.dataset.targets
        auc_valid, pcc_valid = get_auc(targets, scores), get_pcc(targets, scores)
        if pcc_valid > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = pcc_valid
            self.training_state['best_epoch'] = epoch_idx
        return auc_valid, pcc_valid

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        return np.hstack([self.predict_step(data_x, **kwargs)
                          for data_x, _ in data_loader])

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        logger.info(f'Loading model from {self.model_path}')

    def test(self, test_loader, mhc_names, **kwargs):
        scores, targets = self.predict(test_loader, valid=True, **kwargs), test_loader.dataset.targets
        auc_all, pcc_all = get_auc(targets, scores), get_pcc(targets, scores)

        # if auc_all > self.training_state['best']:
        #     self.save_model()
        #     self.training_state['best'] = auc_all

        targets, scores, metrics = np.asarray(targets), np.asarray(scores), []
        mhc_groups, auc, pcc, srcc = get_group_metrics(mhc_names, targets, scores, reduce=False)
        for mhc_name_, auc_, pcc_, srcc_ in zip(mhc_groups, auc, pcc, srcc):
            t_ = targets[mhc_names == mhc_name_]
            metrics.append((auc_, pcc_, srcc_))
        metrics = np.mean(metrics, axis=0)

        auc_group = metrics[0]
        pcc_group = metrics[1]
        srcc_group = metrics[2]
        return auc_all, auc_group, pcc_group, srcc_group
