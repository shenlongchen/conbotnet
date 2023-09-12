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
from conbotnet.models_fine_tuning import ModelFineTuning
from conbotnet.models_binding import ModelBinding

from conbotnet.BoTNet import LinConBoTNet, BinConBoTNet
from conbotnet.evaluation import output_res, CUTOFF


# from data_processing.utils import output_res, CUTOFF


def train(model, data_cnf, model_cnf, train_data, test_data=None, data_group_name=None, cv_id=None, cv_=None,
          valid_data=None, random_state=2023):
    logger.info(f'Start training model {model.model_path}')
    if valid_data is None:
        train_data, valid_data = train_test_split(train_data, test_size=data_cnf.get('valid', 1000),
                                                  random_state=random_state)
    train_loader = DataLoader(EOMHCIIDataset(train_data, **model_cnf['padding']),
                              batch_size=model_cnf['train']['batch_size'], shuffle=True)
    valid_loader = DataLoader(EOMHCIIDataset(valid_data, **model_cnf['padding']),
                              batch_size=model_cnf['valid']['batch_size'])

    if test_data is not None:
        test_loader = DataLoader(EOMHCIIDataset(test_data, **model_cnf['padding']),
                                 batch_size=model_cnf['test']['batch_size'])
        model.train(train_loader, valid_loader, test_loader, data_group_name, cv_id, cv_, **model_cnf['train'])
    else:
        model.train(train_loader, valid_loader, **model_cnf['train'])
    logger.info(f'Finish training model {model.model_path}')


def test(model, model_cnf, test_data):
    data_loader = DataLoader(EOMHCIIDataset(test_data, **model_cnf['padding']),
                             batch_size=model_cnf['test']['batch_size'])
    return model.predict(data_loader)


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="config/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="config/conbotnet_fine_tuning.yaml")
@click.option('--mode', type=click.Choice(('train', 'eval', '5cv', 'lomo', 'binding', 'seq2logo')),
              default='train')
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=20)
@click.option('-c', '--continue', 'continue_train', is_flag=True)
@click.option('-a', '--allele', default='DRB1_0101')
def main(data_cnf, model_cnf, mode, continue_train, start_id, num_models, allele):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_name = model_cnf['name']
    logger.info(f'Model Name: {model_name}')
    model_path = Path(model_cnf['path']) / f'{model_name}.pt'
    res_path = Path(data_cnf['results']) / f'{model_name}'
    model_cnf.setdefault('ensemble', 20)
    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
    get_data_fn = partial(get_data, mhc_name_seq=mhc_name_seq)
    if mode is None or mode == 'train' or mode == 'eval':
        train_data = get_data_fn(data_cnf['train']) if mode is None or mode == 'train' else None
        valid_data = get_data_fn(data_cnf['valid']) if train_data is not None and 'valid' in data_cnf else None
        if mode is None or mode == 'eval' or mode == 'train':
            test_data = get_data_fn(data_cnf['test'])
            test_group_name, test_truth = [x[0] for x in test_data], [x[-1] for x in test_data]
        else:
            test_data = test_group_name = test_truth = None
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            model = ModelFineTuning(LinConBoTNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'),
                                    **model_cnf['model'])
            if train_data is not None:
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data, valid_data=valid_data,
                          test_data=test_data, data_group_name=test_group_name)

            if test_data is not None:
                scores_list.append(test(model, model_cnf, test_data=test_data))
                output_res(test_group_name, test_truth, np.mean(scores_list, axis=0),
                           res_path.with_name(f'{res_path.stem}-train-eval'))
    elif mode == '5cv':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        data_group_name, data_truth = [x[0] for x in data], [x[-1] for x in data]
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        assert len(data) == len(cv_id)
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            scores_ = np.empty(len(data), dtype=np.float32)
            for cv_ in range(5):
                train_data, test_data = data[cv_id != cv_], data[cv_id == cv_]
                model = ModelFineTuning(LinConBoTNet,
                                        model_path=model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}'),
                                        **model_cnf['model'])
                if not continue_train or not model.model_path.exists():
                    train(model, data_cnf, model_cnf, train_data=train_data, test_data=test_data,
                          data_group_name=data_group_name, cv_id=cv_id, cv_=cv_)

                scores_[cv_id == cv_] = test(model, model_cnf, test_data=test_data)
            scores_list.append(scores_)
            output_res(data_group_name, data_truth, np.mean(scores_list, axis=0),
                       res_path.with_name(f'{res_path.stem}-5CV'))
    elif mode == 'lomo':
        data = np.asarray(get_data_fn(data_cnf['train']), dtype=object)
        with open(data_cnf['cv_id']) as fp:
            cv_id = np.asarray([int(line) for line in fp])
        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            group_names, group_names_, truth_, scores_ = np.asarray([x[0] for x in data]), [], [], []
            for name_ in sorted(set(group_names)):
                train_data, train_cv_id = data[group_names != name_], cv_id[group_names != name_]
                test_data, test_cv_id = data[group_names == name_], cv_id[group_names == name_]
                if len(test_data) > 30 and len([x[-1] for x in test_data if x[-1] >= CUTOFF]) >= 3:
                    for cv_ in range(5):
                        model = ModelFineTuning(LinConBoTNet,
                                                model_path=model_path.with_stem(
                                                    F'{model_path.stem}-{name_}-{model_id}-CV{cv_}'),
                                                **model_cnf['model'])
                        if not model.model_path.exists() or not continue_train:
                            train(model, data_cnf, model_cnf, train_data[train_cv_id != cv_])

                        test_data_ = test_data[test_cv_id == cv_]
                        group_names_ += [x[0] for x in test_data_]
                        truth_ += [x[-1] for x in test_data_]
                        scores_ += test(model, model_cnf, test_data_).tolist()
            scores_list.append(scores_)
            output_res(group_names_, truth_, np.mean(scores_list, axis=0), res_path.with_name(f'{res_path.stem}-LOMO'))
    elif mode == 'binding':
        model_cnf['padding'] = model_cnf['binding']
        data_list = get_binding_data(data_cnf['binding'], mhc_name_seq)

        (core_pos, scores), correct = get_binding_core(data_list, model_cnf, model_path, start_id, num_models), 0
        for d, core_pos_, scores_ in zip(data_list, core_pos, scores):
            (pdb, mhc_name, core), peptide_seq = d[0], d[1]
            core_ = peptide_seq[core_pos_: core_pos_ + 9]
            print(pdb, mhc_name, peptide_seq, core, core_, core == core_)
            if core != core_:
                for i, s in enumerate(scores_[:len(peptide_seq) - len(core) + 1]):
                    print(peptide_seq[i: i + len(core)], s)
            correct += core_ == core
        logger.info(f'The number of correct prediction is {correct}.')
    elif mode == 'seq2logo':
        assert allele in mhc_name_seq
        data_list = get_seq2logo_data(data_cnf['seq2logo'], allele, mhc_name_seq[allele])

        scores_list = []
        for model_id in range(start_id, start_id + num_models):
            model = ModelBinding(BinConBoTNet,
                                 model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'),
                                 pooling=False, **model_cnf['model'])
            scores_list.append(test(model, model_cnf, data_list))
        scores = np.mean(scores_list, axis=0)
        s_, p_ = scores.max(axis=1), scores.argmax(axis=1)
        res_path = Path(data_cnf['results_logos'])
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(res_path.joinpath(f'{res_path.stem}-{allele}-{start_id}_{start_id + num_models}.txt'), 'w') as fp:
            for k in (-s_).argsort()[:int(0.01 * len(s_))]:
                print(data_list[k][1][p_[k]: p_[k] + 9], file=fp)


def get_binding_core(data_list, model_cnf, model_path, start_id, num_models):
    scores_list = []
    for model_id in range(start_id, start_id + num_models):
        model = ModelBinding(BinConBoTNet, model_path=model_path.with_stem(f'{model_path.stem}-{model_id}'),
                             pooling=False, **model_cnf['model'])
        scores_list.append(test(model, model_cnf, data_list))
    return (scores := np.mean(scores_list, axis=0)).argmax(-1), scores


if __name__ == '__main__':
    main()
