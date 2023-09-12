# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/15
@Auth ： shenlongchen
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import click
import numpy as np
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from logzero import logger

from conbotnet.data_utils import *
from conbotnet.datasets import EOMHCIIDataset
from conbotnet.models_pretrain import ModelPretrain
from conbotnet.BoTNet import SupConBoTNet
from conbotnet.evaluation import CUTOFF


def train(model, data_cnf, model_cnf, train_data, valid_data=None, random_state=2023):
    logger.info(f'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(train_data, test_size=data_cnf.get('valid', 1000),
                                                  random_state=random_state)
    train_loader = DataLoader(EOMHCIIDataset(train_data, **model_cnf['padding']),
                              batch_size=model_cnf['train']['batch_size'], shuffle=True)
    valid_loader = DataLoader(EOMHCIIDataset(valid_data, **model_cnf['padding']),
                              batch_size=model_cnf['valid']['batch_size'])

    model.train(train_loader, valid_loader, **model_cnf['train'])
    logger.info(f'Finish training model {model.model_path}')


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="config/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="config/conbotnet_pretrain.yaml")
@click.option('--mode', type=click.Choice(('train', '5cv', 'lomo')), default='5cv')
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=20)
@click.option('-c', '--continue', 'continue_train', is_flag=True)
def main(data_cnf, model_cnf, mode, continue_train, start_id, num_models):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']
    logger.info(f'Model Name: {model_name}')
    model_path = Path(model_cnf['path']) / f'{model_name}.pt'
    model_cnf.setdefault('ensemble', 20)
    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
    get_data_fn = partial(get_data, mhc_name_seq=mhc_name_seq)

    if mode is None or mode == 'train':
        train_data = get_data_fn(data_cnf['pretrain']) if mode is None or mode == 'train' else None
        valid_data = get_data_fn(data_cnf['valid']) if train_data is not None and 'valid' in data_cnf else None

        for model_id in range(start_id, start_id + num_models):
            model = ModelPretrain(SupConBoTNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'),
                                  **model_cnf['model'])
            if train_data is not None:
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data)

    elif mode == '5cv':
        data = np.asarray(get_data_fn(data_cnf['pretrain']), dtype=object)
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        assert len(data) == len(cv_id)

        for model_id in range(start_id, start_id + num_models):
            for cv_ in range(5):
                train_data, test_data = data[cv_id != cv_], data[cv_id == cv_]
                model = ModelPretrain(SupConBoTNet,
                                      model_path=model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}'),
                                      **model_cnf['model'])
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data)

    elif mode == 'lomo':
        data = np.asarray(get_data_fn(data_cnf['pretrain']), dtype=object)
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])

        for model_id in range(start_id, start_id + num_models):
            group_names, group_names_, truth_, scores_ = np.asarray([x[0] for x in data]), [], [], []
            for name_ in sorted(set(group_names)):
                train_data, train_cv_id = data[group_names != name_], cv_id[group_names != name_]
                test_data, test_cv_id = data[group_names == name_], cv_id[group_names == name_]
                if len(test_data) > 30 and len([x[-1] for x in test_data if x[-1] >= CUTOFF]) >= 3:
                    for cv_ in range(5):
                        model = ModelPretrain(SupConBoTNet,
                                              model_path=model_path.with_stem(
                                                 F'{model_path.stem}-{name_}-{model_id}-CV{cv_}'),
                                              **model_cnf['model'])
                        if not continue_train or not model.model_path.exists():
                            train(model, data_cnf, model_cnf, train_data[train_cv_id != cv_])


if __name__ == '__main__':
    main()
