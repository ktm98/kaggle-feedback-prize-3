import ast
import copy
import gc
import itertools
import json
import math
import os
import pickle
import random
import re
import string
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy as sp
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

import yaml
import shutil
from attrdict import AttrDict
import logzero
import optuna
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils import get_score, get_args, convert_dot_dict
from engine import train_loop
from process import process

# sys.path.append('./gramformer')
sys.path.append('./base_exp')


def sweep(sweep_config):


    def objective(trial):
        def parse_dict(d):
            d = dict(d)
            dic = {}
            for k, v in d.items():
                if not hasattr(v, "__iter__"):
                    dic[k] = v

                elif 'distribution' in v:

                    if v['distribution'] == 'categorical':
                        dic[k] = getattr(trial, f'suggest_{v["distribution"]}')(k, v['values'])
                    else:
                        dic[k] = getattr(trial, f'suggest_{v["distribution"]}')(k, v['min'], v['max'])
                elif type(v) == dict:
                    dic[k] = parse_dict(v)

                else:
                    dic[k] = v
  
            return dic

        CFG = parse_dict(sweep_config)
        CFG = AttrDict(CFG)
        CFG._setattr('_sequence_type', list)

        os.makedirs(CFG.output_dir, exist_ok=True)
        with open(CFG.output_dir + f'config_{trial.number}.yml', 'w') as f:
            yaml.dump(dict(CFG), f)

        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        train_df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
        # test_df = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
        # submission_df = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')
        train_df = process(train_df)
        target_cols = CFG.target_cols
        if CFG.use_unique_token:
            train_df['n_unique_tokens'] = train_df['full_text'].apply(lambda x: len(set(x.split()))//100)
            target_cols = CFG.target_cols + ['n_unique_tokens']

        Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[target_cols])):
            train_df.loc[val_index, 'fold'] = int(n)
        train_df['fold'] = train_df['fold'].astype(int)

        tokenizer = AutoTokenizer.from_pretrained(CFG.model)


        # tokenizer.save_pretrained(CFG.output_dir+'tokenizer/')
        CFG.tokenizer = tokenizer

        lengths = []
        tk0 = tqdm(train_df['full_text'].fillna("").values, total=len(train_df))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            lengths.append(length)
        CFG.max_len = max(lengths) + 2 # cls & sep & sep
        # LOGGER.info(f"max_len: {CFG.max_len}")

        def get_result(oof_df):
            labels = oof_df[CFG.target_cols].values
            preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
            score, scores = get_score(labels, preds)
            # LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
            
            return score, scores

        if CFG.train:
            oof_df = pd.DataFrame()
            for fold in range(CFG.n_fold):
                if fold in CFG.trn_fold:
                    _oof_df = train_loop(train_df, fold, CFG)
                    oof_df = pd.concat([oof_df, _oof_df])
                    # LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(_oof_df)
            oof_df = oof_df.reset_index(drop=True)
            # LOGGER.info(f"========== CV ==========")
            score, scores = get_result(oof_df)
        return score

    def save_log(study, trial):
        LOGGER.info(f"Best trial: {study.best_trial.number}")
        LOGGER.info(f"Best trial params: {study.best_trial.params}")
        LOGGER.info(f'Best score: {study.best_value}')


    os.makedirs(sweep_config['output_dir'], exist_ok=True)
    LOGGER = logzero.setup_logger(
        logfile=sweep_config['output_dir']+'train.log', level=20, fileLoglevel=20)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective,
     n_trials=100,
    #   timeout=60*60*24*0.5,
      callbacks=[save_log]
     )

    print('Best score ', study.best_value)
    print('Best params ', study.best_params)
    print('Best trial ', study.best_trial.number)
    LOGGER.info(f'Best score {study.best_value}')
    LOGGER.info(f'Best params {study.best_params}')
    LOGGER.info(f'Best trial {study.best_trial.number}')
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(sweep_config['output_dir']+'importance.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YAMLありの例')
    parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
    # parser.add_argument('--sweep_config', type=str, help='sweepのconfigファイル')  # 
    # parser.add_argument('--debug', action='store_true', help='引数が指定されたらTrue、指定されなかったらFalse')
    args = parser.parse_args()
    path = args.config_path

    with open(path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    sweep(sweep_config)