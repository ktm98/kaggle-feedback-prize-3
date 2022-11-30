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
import yaml
from attrdict import AttrDict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          DataCollatorWithPadding,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from utils import convert_dot_dict, get_args, get_score

from dataset import TestDataset
from engine import inference_fn, train_loop
from mlm import load_2021_and_remove_duplicates
from model import CustomModel
from process import process

os.environ['TOKENIZERS_PARALLELISM']='false'

sys.path.append('./base_exp')

def main():
    args = get_args()
    exp_dir = args.config_path
    exp_path = os.path.join('/root/result', exp_dir)
    with open(os.path.join(exp_path, 'params.yml'), 'r') as f:
        CFG = yaml.safe_load(f)
    debug = args.debug

    CFG = AttrDict(CFG)
    # print(cfg)
    # https://github.com/bcj/AttrDict/issues/34
    CFG._setattr('_sequence_type', list)

    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.output_dir+'tokenizer/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_df = load_2021_and_remove_duplicates()
    test_df = process(test_df)

    test_df['tokenize_length'] = [len(CFG.tokenizer(text)['input_ids']) for text in test_df['full_text'].values]
    test_df = test_df.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    test_dataset = TestDataset(CFG, test_df)
    test_loader = DataLoader(test_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    predictions = []
    for fold in CFG.trn_fold:
        model = CustomModel(CFG, config_path=CFG.output_dir+'config.pth', pretrained=False)
        state = torch.load(CFG.output_dir+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                        map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        predictions.append(prediction)
        test_df[CFG.target_cols] = prediction
        test_df.to_csv(CFG.output_dir+f"2021_pl_fold{fold}.csv", index=False)
        del model, state, prediction; gc.collect()
        torch.cuda.empty_cache()
    # predictions = np.mean(predictions, axis=0)

    # test_df[CFG.target_cols] = predictions

    # test_df.to_csv(CFG.output_dir+'2021_pl.csv', index=False)
    print(test_df)

if __name__ == '__main__':
    main()