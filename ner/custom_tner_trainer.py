#!/usr/bin/env python3

"""Training scripts for finetuning TNER.

This script is based on the following scripts (under MIT license):
    * https://github.com/asahi417/tner/blob/master/tner/ner_trainer.py
	* https://github.com/asahi417/tner/blob/master/tner/get_dataset.py

It modifies the original Trainer class , so that instead of initializing the
model from a pre-trained BERT-like model, it starts with an already trained
TNER models.

The `load_conll_format_file` is just copied (and simplified) from the original
script, so it can be used in the custom `get_dataset` fuction than uses the
provided `label2id` dictionary.
"""

import argparse
import os
import json
import logging
import random
import gc
import shutil
from glob import glob
from os.path import join as pj
from typing import List, Dict
from itertools import product
from distutils.dir_util import copy_tree
from unicodedata import normalize
from itertools import chain

import torch
import transformers

from tner import TransformersNER
from tner.util import json_save, json_load, get_random_string


def load_conll_format_file(data_path: str, label2id: Dict = None):
    """ load dataset from local IOB format file

    @param data_path: path to iob file
    @param label2id: [optional] dictionary of label2id (generate from dataset as default )
    @return: (data, label2id)
        - data: a dictionary of {"tokens": [list of tokens], "tags": [list of tags]}
    """
    inputs, labels, seen_entity = [], [], []
    with open(data_path, 'r') as f:
        sentence, entity = [], []
        for n, line_raw in enumerate(f):
            line = normalize('NFKD', line_raw).strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(sentence) != 0:
                    assert len(sentence) == len(entity)
                    inputs.append(sentence)
                    labels.append(entity)
                    sentence, entity = [], []
            else:
                ls = line.split()
                if len(ls) < 2:
                    if line_raw.startswith('O'):
                        logging.warning(f'skip {ls} (line {n} of file {data_path}): '
                                        f'missing token (should be word and tag separated by '
                                        f'a half-space, eg. `London B-LOC`)')
                        continue
                    else:
                        ls = ['', ls[0]]
                # Examples could have no label for mode = "test"
                word, tag = ls[0], ls[-1]
                sentence.append(word)
                entity.append(tag)

        if len(sentence) != 0:
            assert len(sentence) == len(entity)
            inputs.append(sentence)
            labels.append(entity)

    all_labels = sorted(list(set(list(chain(*labels)))))
    labels_not_found = [i for i in all_labels if i not in label2id]
    if len(labels_not_found) > 0:
        logging.warning(f'found entities not in the label2id (label2id was updated):\n\t - {labels_not_found}')
        label2id.update({i: len(label2id) + n for n, i in enumerate(labels_not_found)})
    assert all(i in label2id for i in all_labels), \
        f"label2id is not covering all the entity \n \t- {label2id} \n \t- {all_labels}"
    keys = label2id.copy().keys()
    for l in keys:
        if l.startswith('B'):
            entity = l[2:]
            if 'I-'+entity not in label2id:
                label2id.update({'I-'+entity: len(label2id)})
                logging.warning(f'found entities without I label2id (label2id was updated):\n\t - {entity}')
    labels = [[label2id[__l] for __l in _l] for _l in labels]
    data = {"tokens": inputs, "tags": labels}
    return data


def get_dataset(path_dict, label2id):
    output = {}
    for split, path in path_dict.items():
        data = load_conll_format_file(path, label2id)
        output[split] = data
    return output


class Trainer:
    """ fine-tuning language model on NER """

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: List or str = None,
                 local_dataset: List or Dict = None,
                 dataset_split: str = 'train',
                 dataset_name: List or str = None,
                 model: TransformersNER = None,
                 crf: bool = False,
                 max_length: int = 128,
                 epoch: int = 10,
                 batch_size: int = 128,
                 lr: float = 1e-4,
                 random_seed: int = 42,
                 gradient_accumulation_steps: int = 1,
                 weight_decay: float = 1e-7,
                 lr_warmup_step_ratio: float = None,
                 max_grad_norm: float = None,
                 disable_log: bool = False,
                 use_auth_token: bool = False,
                 config_file: str = 'trainer_config.json'):
        """ fine-tuning language model on NER

        @param checkpoint_dir: directly to save model weight and other information
        @param dataset: dataset name (or a list of it) on huggingface tner organization
            (eg. "tner/conll2003", ["tner/conll2003", "tner/ontonotes5"]]
            see https://huggingface.co/datasets?search=tner for full dataset list
        @param local_dataset: a dictionary (or a list) of paths to local BIO files eg.
            {"train": "examples/local_dataset_sample/train.txt", "test": "examples/local_dataset_sample/test.txt"}
        @param dataset_split: [optional] dataset split to be used ('train' as default)
        @param dataset_name: [optional] data name of huggingface dataset (should be same length as the `dataset`)
        @param model: model name of underlying language model (huggingface model)
        @param crf: use CRF on top of output embedding
        @param max_length: max length of language model
        @param epoch: the number of epoch
        @param batch_size: batch size
        @param lr: learning rate
        @param random_seed: random seed
        @param gradient_accumulation_steps: the number of gradient accumulation
        @param weight_decay: coefficient of weight decay
        @param lr_warmup_step_ratio: linear warmup ratio of learning rate
            eg) if it's 0.3, the learning rate will warmup linearly till 30% of the total step (no decay after all)
        @param max_grad_norm: norm for gradient clipping
        @param disable_log: [optional] disabling logging
        @param use_auth_token: [optional] Huggingface transformers argument of `use_auth_token`
        @param config_file: [optional] name of config file
        """

        # validate config
        self.scheduler = None
        self.optimizer = None
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.current_epoch = 0
        self.config = dict(
            dataset=dataset, dataset_split=dataset_split,
            dataset_name=dataset_name, local_dataset=local_dataset,
            model=model.model_name if model is not None else None,
            crf=crf, max_length=max_length, epoch=epoch, batch_size=batch_size,
            lr=lr, random_seed=random_seed, gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay, lr_warmup_step_ratio=lr_warmup_step_ratio, max_grad_norm=max_grad_norm
        )

        # check local directly whether in progress checkpoints exist
        for e in sorted([int(i.split('epoch_')[-1]) for i in glob(pj(self.checkpoint_dir, 'epoch_*'))], reverse=True):
            if not os.path.exists(pj(self.checkpoint_dir, "optimizers", f"optimizer.{e}.pt")):
                continue
            model_path = pj(self.checkpoint_dir, f'epoch_{e}')
            try:
                logging.info(f'load checkpoint from {model_path}')

                config = json_load(pj(self.checkpoint_dir, config_file))
                self.model = TransformersNER(
                    model_path,
                    max_length=config['max_length'],
                    crf=config['crf'],
                    use_auth_token=use_auth_token
                )
                self.current_epoch = e
                assert self.current_epoch <= config['epoch'],\
                    f'model training is over {self.checkpoint_dir}: {self.current_epoch} == {config["epoch"]}'
                logging.warning(f'config is overwritten by {model_path}')
                self.config = config
            except Exception:
                logging.exception(f'error at loading checkpoint {model_path}')

        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info(f'\t * {k}: {v}')

        # Here, the call of the original `get_dataset` function is replaced by
        # a custom function from this file.
        data = get_dataset(
            self.config['local_dataset'],
            label2id=self.model.label2id)
        assert self.config['dataset_split'] in data, f"split {self.config['dataset_split']} is not in {data.keys()}"
        self.dataset = data[self.config['dataset_split']]
        self.step_per_epoch = int(
            len(self.dataset['tokens']) / self.config['batch_size'] / self.config['gradient_accumulation_steps']
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(pj(self.checkpoint_dir, config_file), 'w') as f:
            json.dump(self.config, f)

        random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])

        if not disable_log:
            # add file handler
            logger = logging.getLogger()
            file_handler = logging.FileHandler(pj(self.checkpoint_dir, 'training.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
            logger.addHandler(file_handler)

    def save(self, current_epoch: int):
        """ save checkpoint

        @param current_epoch: checkpoint is saved as "epoch_[current_epoch]"
        """
        # save model
        save_dir = pj(self.checkpoint_dir, f'epoch_{current_epoch + 1}')
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f'model saving at {save_dir}')
        self.model.save(save_dir)
        # save optimizer
        save_dir_opt = pj(self.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch + 1}.pt')
        os.makedirs(os.path.dirname(save_dir_opt), exist_ok=True)
        # Fix the memory error
        logging.info(f'optimizer saving at {save_dir_opt}')
        if self.scheduler is not None:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_dir_opt)
        else:
            torch.save({'optimizer_state_dict': self.optimizer.state_dict()}, save_dir_opt)
        logging.info('remove old optimizer files')
        if os.path.exists(pj(self.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch}.pt')):
            os.remove(pj(self.checkpoint_dir, 'optimizers', f'optimizer.{current_epoch}.pt'))

    def train(self,
              epoch_save: int = 1,
              epoch_partial: int = None,
              optimizer_on_cpu: bool = False):
        """ train model

        @param epoch_save: interval of epoch to save intermediate checkpoint (every single epoch as default)
        @param epoch_partial: epoch to stop training in the middle of full training
        @param optimizer_on_cpu: put optimizer on CPU to save memory of GPU
        """
        logging.info('dataset preprocessing')
        self.model.train()
        self.setup_optimizer(optimizer_on_cpu)
        assert self.current_epoch != self.config['epoch'], 'training is over'
        assert len(self.dataset['tokens']) >= self.config['batch_size'],\
            f"batch size should be less than the dataset ({len(self.dataset['tokens'])})"
        loader = self.model.get_data_loader(
            inputs=self.dataset['tokens'],
            labels=self.dataset['tags'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True,
            cache_file_feature=pj(self.checkpoint_dir, "cache", "encoded_feature.pkl")
        )
        logging.info('start model training')
        interval = 50
        for e in range(self.current_epoch, self.config['epoch']):  # loop over the epoch
            total_loss = []
            self.optimizer.zero_grad()
            for n, encode in enumerate(loader):
                loss = self.model.encode_to_loss(encode)
                loss.backward()
                if self.config['max_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config['max_grad_norm'])
                total_loss.append(loss.cpu().item())
                if (n + 1) % self.config['gradient_accumulation_steps'] != 0:
                    continue
                # optimizer update
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                if len(total_loss) % interval == 0:
                    _tmp_loss = round(sum(total_loss) / len(total_loss), 2)
                    lr = self.optimizer.param_groups[0]['lr']
                    logging.info(f"\t * global step {len(total_loss)}: loss: {_tmp_loss}, lr: {lr}")
            self.optimizer.zero_grad()
            _tmp_loss = round(sum(total_loss) / len(total_loss), 2)
            lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"[epoch {e}/{self.config['epoch']}] average loss: {_tmp_loss}, lr: {lr}")
            if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != 0:
                self.save(e)
            if epoch_partial is not None and (e + 1) == epoch_partial:
                break
        self.save(e)
        self.current_epoch = e + 1
        logging.info(f'complete training: model ckpt was saved at {self.checkpoint_dir}')

    def setup_optimizer(self, optimizer_on_cpu):
        """ setup optimizer and scheduler

        @param optimizer_on_cpu: put optimizer on CPU to save memory of GPU
        """
        # optimizer
        if self.config['weight_decay'] is not None and self.config['weight_decay'] != 0:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.config['weight_decay']},
                {"params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0}]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'])
        else:
            self.optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.config['lr'])
        if self.config['lr_warmup_step_ratio'] is not None:
            total_step = self.step_per_epoch * self.config['epoch']
            num_warmup_steps = int(total_step * self.config['lr_warmup_step_ratio'])
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_step)

        # resume fine-tuning
        if self.current_epoch is not None and self.current_epoch != 0:
            path = pj(self.checkpoint_dir, "optimizers", f'optimizer.{self.current_epoch}.pt')
            logging.info(f'load optimizer from {path}')
            device = 'cpu' if optimizer_on_cpu == 1 else self.model.device
            logging.info(f'optimizer is loading on {device}')
            stats = torch.load(path, map_location=torch.device(device))
            self.optimizer.load_state_dict(stats['optimizer_state_dict'])
            if self.scheduler is not None:
                logging.info(f'load scheduler from {path}')
                self.scheduler.load_state_dict(stats['scheduler_state_dict'])
            del stats
            gc.collect()


class GridSearcher:
    """ fine-tuning language model on NER with grid search over different configs
    tunable parameters with examples:
        - gradient_accumulation_steps=[4, 8]
        - crf=[True, False]
        - lr=[1e-4, 1e-3]
        - weight_decay=[1e-5, 1e-6]
        - random_seed=[0, 1, 2]
        - lr_warmup_step_ratio=[None, 0.1, 0.3]
        - max_grad_norm=[None, 10]
    """

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: List or str = None,
                 local_dataset: List or Dict = None,
                 dataset_split_train: str = 'train',
                 dataset_split_valid: str = 'validation',
                 dataset_name: List or str = None,
                 model: str = 'roberta-large',
                 epoch: int = 10,
                 epoch_partial: int = 5,
                 n_max_config: int = 5,
                 max_length: int = 128,
                 max_length_eval: int = 128,
                 batch_size: int = 32,
                 batch_size_eval: int = 16,
                 gradient_accumulation_steps: List or int = 1,
                 crf: List or bool = True,
                 lr: List or float = 1e-4,
                 weight_decay: List or float = None,
                 random_seed: List or int = 0,
                 lr_warmup_step_ratio: List or int = None,
                 max_grad_norm: List or float = None,
                 validation_metric: str = 'micro/f1',
                 use_auth_token: bool = False):
        """ fine-tuning language model on NER with grid search over different configs

        @param checkpoint_dir: directly to save model weight and other information
        @param dataset: dataset name (or a list of it) on huggingface tner organization
            (eg. "tner/conll2003", ["tner/conll2003", "tner/ontonotes5"]]
            see https://huggingface.co/datasets?search=tner for full dataset list
        @param local_dataset: a dictionary (or a list) of paths to local BIO files eg.
            {"train": "examples/local_dataset_sample/train.txt", "test": "examples/local_dataset_sample/test.txt"}
        @param dataset_split_train: [optional] dataset split to be used for training ('train' as default)
        @param dataset_split_valid: [optional] dataset split to be used for validation ('valid' as default)
        @param dataset_name: [optional] data name of huggingface dataset (should be same length as the `datasets`)
        @param model: model name of underlying language model (huggingface model)
        @param epoch: the number of epoch
        @param epoch_partial: the number of epoch for 1st phase search
        @param n_max_config: the number of configs to run 2nd phase search
        @param max_length: max length of language model
        @param max_length_eval: max length of language model at evaluation
        @param batch_size: batch size
        @param batch_size_eval: batch size at evaluation
        @param gradient_accumulation_steps: the number of gradient accumulation
        @param crf: use CRF on top of output embedding
        @param lr: learning rate
        @param weight_decay: coefficient for weight decay
        @param random_seed: random seed
        @param lr_warmup_step_ratio: linear warmup ratio of learning rate
            eg) if it's 0.3, the learning rate will warmup linearly till 30% of the total step (no decay after all)
        @param max_grad_norm: norm for gradient clipping
        @param validation_metric: metric to be used for validation
        @param use_auth_token: [optional] Huggingface transformers argument of `use_auth_token`
        """

        self.checkpoint_dir = checkpoint_dir
        self.epoch_partial = epoch_partial
        self.batch_size_eval = batch_size_eval
        self.n_max_config = n_max_config

        # evaluation configs
        self.eval_config = {
            'max_length_eval': max_length_eval,
            'metric': validation_metric,
            'dataset_split_valid': dataset_split_valid
        }
        # static configs
        self.static_config = {
            'dataset': dataset,
            'local_dataset': local_dataset,
            'dataset_name': dataset_name,
            'dataset_split': dataset_split_train,
            'model': model,
            'batch_size': batch_size,
            'epoch': epoch,
            'max_length': max_length,
            'use_auth_token': use_auth_token
        }

        # dynamic config
        def to_list(_val):
            if type(_val) != list:
                return [_val]
            assert len(_val) == len(set(_val)), _val
            if None in _val:
                _val.pop(_val.index(None))
                return [None] + sorted(_val, reverse=True)
            return sorted(_val, reverse=True)

        self.dynamic_config = {
            'lr': to_list(lr),
            'crf': to_list(crf),
            'random_seed': to_list(random_seed),
            'weight_decay': to_list(weight_decay),
            'lr_warmup_step_ratio': to_list(lr_warmup_step_ratio),
            'max_grad_norm': to_list(max_grad_norm),
            'gradient_accumulation_steps': to_list(gradient_accumulation_steps)
        }

        self.all_dynamic_configs = list(product(
            self.dynamic_config['lr'],
            self.dynamic_config['crf'],
            self.dynamic_config['random_seed'],
            self.dynamic_config['weight_decay'],
            self.dynamic_config['lr_warmup_step_ratio'],
            self.dynamic_config['max_grad_norm'],
            self.dynamic_config['gradient_accumulation_steps'],
        ))

    def train(self, optimizer_on_cpu: bool = False):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # sanity check
        for _f, c in zip(['config_static', 'config_dynamic.json', 'config_eval.json'],
                         [self.static_config, self.dynamic_config, self.eval_config]):
            if os.path.exists(pj(self.checkpoint_dir, _f)):
                tmp = json_load(pj(self.checkpoint_dir, _f))
                tmp_v = [tmp[k] for k in sorted(tmp.keys())]
                _tmp_v = [c[k] for k in sorted(tmp.keys())]
                assert tmp_v == _tmp_v, f'{str(tmp_v)}\n not matched \n{str(_tmp_v)}'
        json_save(self.static_config, pj(self.checkpoint_dir, 'config_static.json'))
        json_save(self.dynamic_config, pj(self.checkpoint_dir, 'config_dynamic.json'))
        json_save(self.eval_config, pj(self.checkpoint_dir, 'config_eval.json'))

        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler(pj(self.checkpoint_dir, 'grid_search.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)
        logging.info(f'INITIALIZE GRID SEARCHER: {len(self.all_dynamic_configs)} configs to try')
        cache_prefix = pj(self.checkpoint_dir, "encoded", f"{self.static_config['model']}.{self.static_config['max_length']}.dev")

        ###########
        # 1st RUN #
        ###########
        checkpoints = []
        ckpt_exist = {}
        for trainer_config in glob(pj(self.checkpoint_dir, 'model_*', 'trainer_config.json')):
            ckpt_exist[os.path.dirname(trainer_config)] = json_load(trainer_config)
        for n, dynamic_config in enumerate(self.all_dynamic_configs):
            logging.info(f'## 1st RUN: Configuration {n}/{len(self.all_dynamic_configs)} ##')
            config = self.static_config.copy()
            tmp_dynamic_config = {'lr': dynamic_config[0], 'crf': dynamic_config[1], 'random_seed': dynamic_config[2],
                                  'weight_decay': dynamic_config[3], 'lr_warmup_step_ratio': dynamic_config[4],
                                  'max_grad_norm': dynamic_config[5], 'gradient_accumulation_steps': dynamic_config[6]}
            config.update(tmp_dynamic_config)
            ex_dynamic_config = [(k_, [v[k] for k in sorted(tmp_dynamic_config.keys())]) for k_, v in ckpt_exist.items()]
            tmp_dynamic_config = [tmp_dynamic_config[k] for k in sorted(tmp_dynamic_config.keys())]
            duplicated_ckpt = [k for k, v in ex_dynamic_config if v == tmp_dynamic_config]

            if len(duplicated_ckpt) == 1:
                checkpoint_dir = duplicated_ckpt[0]
            elif len(duplicated_ckpt) == 0:
                ckpt_name_exist = [os.path.basename(k).replace('model_', '') for k in ckpt_exist.keys()]
                ckpt_name_made = [os.path.basename(c).replace('model_', '') for c in checkpoints]
                model_ckpt = get_random_string(exclude=ckpt_name_exist + ckpt_name_made)
                checkpoint_dir = pj(self.checkpoint_dir, f'model_{model_ckpt}')
            else:
                raise ValueError(f'duplicated checkpoints are found: \n {duplicated_ckpt}')

            if not os.path.exists(pj(checkpoint_dir, f'epoch_{self.epoch_partial}')):
                config_copy = config.copy()
                config_copy['model'] = TransformersNER(config_copy['model'])
                trainer = Trainer(checkpoint_dir=checkpoint_dir, disable_log=True, **config_copy)
                trainer.train(epoch_partial=self.epoch_partial, epoch_save=1, optimizer_on_cpu=optimizer_on_cpu)
            checkpoints.append(checkpoint_dir)

        path_to_metric_1st = pj(self.checkpoint_dir, 'metric.1st.json')
        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info(f'## 1st RUN (EVAL): Configuration {n}/{len(checkpoints)} ##')
            checkpoint_dir_model = pj(checkpoint_dir, f'epoch_{self.epoch_partial}')
            metric, tmp_metric = self.validate_model(checkpoint_dir_model, cache_prefix)
            json_save(metric, pj(checkpoint_dir_model, "eval", "metric.json"))
            metrics[checkpoint_dir_model] = tmp_metric[self.eval_config['metric']]
        metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        json_save(metrics, path_to_metric_1st)

        logging.info('1st RUN RESULTS')
        for n, (k, v) in enumerate(metrics):
            logging.info(f'\t * rank: {n} | metric: {round(v, 3)} | model: {k} |')

        if self.epoch_partial == self.static_config['epoch']:
            logging.info('No 2nd phase as epoch_partial == epoch')
            return

        ###########
        # 2nd RUN #
        ###########
        metrics = metrics[:min(len(metrics), self.n_max_config)]
        checkpoints = []
        for n, (checkpoint_dir_model, _metric) in enumerate(metrics):
            logging.info(f'## 2nd RUN: Configuration {n}/{len(metrics)}: {_metric}')
            model_ckpt = os.path.dirname(checkpoint_dir_model)
            if not os.path.exists(pj(model_ckpt, f"epoch_{self.static_config['epoch']}")):
                trainer = Trainer(checkpoint_dir=model_ckpt, disable_log=True)
                trainer.train(epoch_save=1, optimizer_on_cpu=optimizer_on_cpu)
            checkpoints.append(model_ckpt)
        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info(f'## 2nd RUN (EVAL): Configuration {n}/{len(checkpoints)} ##')
            for checkpoint_dir_model in sorted(glob(pj(checkpoint_dir, 'epoch_*'))):
                metric, tmp_metric = self.validate_model(checkpoint_dir_model, cache_prefix)
                json_save(metric, pj(checkpoint_dir_model, 'eval', 'metric.json'))
                metrics[checkpoint_dir_model] = tmp_metric[self.eval_config['metric']]
        metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        logging.info(f'2nd RUN RESULTS: \n{str(metrics)}')
        for n, (k, v) in enumerate(metrics):
            logging.info(f'\t * rank: {n} | metric: {round(v, 3)} | model: {k} |')
        json_save(metrics, pj(self.checkpoint_dir, 'metric.2nd.json'))

        best_model_ckpt, best_metric_score = metrics[0]
        epoch = int(best_model_ckpt.split(os.path.sep)[-1].replace('epoch_', ''))
        best_model_dir = os.path.dirname(best_model_ckpt)
        with open(pj(best_model_dir, 'trainer_config.json')) as f:
            config = json.load(f)

        if epoch == self.static_config['epoch']:
            ###########
            # 3rd RUN #
            ###########
            logging.info(f'## 3rd RUN: target model: {best_model_dir} (metric: {best_metric_score}) ##')
            metrics = [[epoch, best_metric_score]]
            while True:
                epoch += 1
                logging.info(f'## 3rd RUN (TRAIN): epoch {epoch} ##')
                config['epoch'] = epoch
                with open(pj(best_model_dir, 'trainer_config.additional_training.json'), 'w') as f:
                    json.dump(config, f)
                checkpoint_dir_model = pj(best_model_dir, f'epoch_{epoch}')
                if not os.path.exists(checkpoint_dir_model):
                    trainer = Trainer(
                        checkpoint_dir=best_model_dir,
                        config_file='trainer_config.additional_training.json',
                        disable_log=True)
                    trainer.train(epoch_save=1, optimizer_on_cpu=optimizer_on_cpu)
                logging.info(f'## 3rd RUN (EVAL): epoch {epoch} ##')

                metric, tmp_metric = self.validate_model(checkpoint_dir_model, cache_prefix)
                tmp_metric_score = tmp_metric[self.eval_config['metric']]
                metrics.append([epoch, tmp_metric_score])
                logging.info(f'\t tmp metric: {tmp_metric_score}')
                if best_metric_score > tmp_metric_score:
                    logging.info('\t finish 3rd phase (no improvement)')
                    break
                else:
                    logging.info(f'\t tmp metric improved the best model ({best_metric_score} --> {tmp_metric_score})')
                    best_metric_score = tmp_metric_score
            logging.info(f'3rd RUN RESULTS: {best_model_dir}')
            for k, v in metrics:
                logging.info(f'\t epoch {k}: {v}')
            json_save(metrics, pj(self.checkpoint_dir, 'metric.3rd.json'))
            config['epoch'] = epoch - 1
            best_model_ckpt = f"{best_model_ckpt.split('epoch_')[0]}epoch_{config['epoch']}"

        copy_tree(best_model_ckpt, pj(self.checkpoint_dir, 'best_model'))
        shutil.rmtree(pj(best_model_ckpt, 'eval'))
        with open(pj(self.checkpoint_dir, 'best_model', 'trainer_config.json'), 'w') as f:
            json.dump(config, f)

    def validate_model(self, checkpoint_dir_model, cache_prefix):
        """ validate model checkpoint """
        if os.path.exists(pj(checkpoint_dir_model, 'eval', 'metric.json')):
            metric = json_load(pj(checkpoint_dir_model, 'eval', 'metric.json'))
        else:
            metric = {}
        if self.eval_config['dataset_split_valid'] in metric:
            tmp_metric = metric[self.eval_config['dataset_split_valid']]
        else:
            tmp_model = TransformersNER(
                checkpoint_dir_model,
                max_length=self.eval_config['max_length_eval'],
                use_auth_token=self.static_config['use_auth_token']
            )
            cache_file_feature = f"{cache_prefix}.{tmp_model.crf_layer is not None}.{self.eval_config['dataset_split_valid']}.pkl"
            cache_file_prediction = pj(checkpoint_dir_model, "eval", f"prediction.{self.eval_config['dataset_split_valid']}.json")
            tmp_metric = tmp_model.evaluate(
                dataset=self.static_config['dataset'],
                local_dataset=self.static_config['local_dataset'],
                dataset_name=self.static_config['dataset_name'],
                batch_size=self.batch_size_eval,
                dataset_split=self.eval_config['dataset_split_valid'],
                cache_file_feature=cache_file_feature,
                cache_file_prediction=cache_file_prediction
            )
            metric[self.eval_config['dataset_split_valid']] = tmp_metric
        return metric, tmp_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', type=str) #, default='finetune_tner/checkpoints')
    parser.add_argument('init_model', type=str) #, default='tner/roberta-large-ontonotes5')
    parser.add_argument('train_data', type=str) #, default='finetune_tner/train.tner')
    parser.add_argument('validation_data', type=str) #, default='finetune_tner/dev.tner')
    parser.add_argument('test_data', type=str) #, default='finetune_tner/test.tner')
    args = parser.parse_args()

    local_dataset = {
        "train": args.train_data,
        "validation": args.validation_data,
        "test": args.test_data
    }

    searcher = GridSearcher(
       checkpoint_dir=args.checkpoint_dir,
       local_dataset=local_dataset,
       model=args.init_model,
       epoch=5,                        # the total epoch (`L` in the figure)
       epoch_partial=1,                # the number of epoch at 1st stage (`M` in the figure)
       n_max_config=3,                 # the number of models to pass to 2nd stage (`K` in the figure)
       batch_size=64,
       gradient_accumulation_steps=[1],
       crf=[True],
       lr=[1e-5, 1e-6],
       weight_decay=[None, 1e-7, 1e-6],
       random_seed=[42],
       lr_warmup_step_ratio=[None],
       max_grad_norm=[5, 10]
    )

    searcher.train()


if __name__ == '__main__':
    main()
