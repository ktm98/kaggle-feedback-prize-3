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

from utils import get_score, get_args, convert_dot_dict
from engine import train_loop
from process import process
from mlm import load_2021_and_remove_duplicates

sys.path.append('./base_exp')

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    args = get_args()
    with open(args.config_path, 'r') as f:
        CFG = yaml.safe_load(f)
    debug = args.debug

    CFG = AttrDict(CFG)
    # print(cfg)
    # https://github.com/bcj/AttrDict/issues/34
    CFG._setattr('_sequence_type', list)

    if hasattr(CFG, 'output_dir'):
        exp_name = CFG.output_dir.split('/')[-2]
    else:
        exp_name = CFG.output_dir.value.split('/')[-2]

    if not debug:
        wandb.init(project='FB3', entity='ktm98',
                   config=CFG, name=f'{exp_name}')
        CFG = AttrDict(convert_dot_dict(dict(wandb.config)))
        CFG._setattr('_sequence_type', list)
        wandb.run.log_code(".")
        print(type(CFG))
    
    os.makedirs(CFG.output_dir, exist_ok=True)

    shutil.copyfile(args.config_path, CFG.output_dir+'params.yml')

    LOGGER = logzero.setup_logger(
        logfile=CFG.output_dir+'train.log', level=20, fileLoglevel=20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_df = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
    test_df = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
    submission_df = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')

    # if CFG.use_pl:
    #     print('using pl')
    #     df = pd.read_csv(os.path.join('/root/result', CFG.pl_exp_name, '2021_pl.csv'))
    #     df = process(df)


    train_df = process(train_df)
    target_cols = CFG.target_cols
    if CFG.use_unique_token:
        train_df['n_unique_tokens'] = train_df['full_text'].apply(lambda x: len(set(x.split()))//100)
        target_cols = CFG.target_cols + ['n_unique_tokens']

    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[target_cols])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)

    # if CFG.use_pl:
    #     df['fold'] = -1
    #     train_df = pd.concat([train_df, df], axis=0).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)

    if CFG.add_new_token:
        new_tokens = ['STUDENT_NAME', 'PROPER_NAME', 'Generic_Name', 'Generic_School', 'LOCATION_NAME', 'OTHER_NAME', 'RESTAURANT_NAME',
                    'TEACHER_NAME', 'STORE_NAME', 'LANGUAGE_NAME', 'Generic_City', '\n']
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})


    tokenizer.save_pretrained(CFG.output_dir+'tokenizer/')
    CFG.tokenizer = tokenizer



    # ====================================================
    # Define max_len
    # ====================================================
    lengths = []
    tk0 = tqdm(train_df['full_text'].fillna("").values, total=len(train_df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    # CFG.max_len = max(lengths) + 2 # cls & sep & sep
    LOGGER.info(f"max_len: {CFG.max_len}")

    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
        
        return score, scores

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train_df, fold, CFG)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        score, scores = get_result(oof_df)
        if CFG.wandb:
            wandb.log({'CV': score, 'scores': scores})
        oof_df.to_csv(CFG.output_dir+'oof_df.csv')

    if CFG.wandb:
        wandb.finish()

if __name__ == '__main__': 
    main()
