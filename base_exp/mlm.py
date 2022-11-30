
import copy
import gc
import json
import math
import os
import pickle
import random
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy as sp
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
import wandb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer,
                            DataCollatorForLanguageModeling, Trainer, TrainingArguments,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


import yaml
import shutil
from attrdict import AttrDict
import logzero

from utils import get_score, get_args, convert_dot_dict
from engine import train_loop
from process import process

sys.path.append('./base_exp')

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def load_2021():
    competition_path = '../input/feedback-prize-2021/'
    train_df = pd.read_csv(competition_path + 'train.csv')
    test_df = pd.read_csv(competition_path + 'sample_submission.csv')
    dfs = [train_df, test_df]
    essay_texts = {}
    for i, phase in enumerate(['train', 'test']):
        base_path = competition_path + phase + '/'


        for filename in os.listdir(base_path):
            with open(base_path + filename) as f:
                text = f.readlines()
                full_text = ' '.join([x for x in text])
                essay_text = ' '.join([x for x in full_text.split()])
            essay_texts[filename[:-4]] = essay_text
        
    df = pd.Series(essay_texts).to_frame().reset_index().rename(columns={'index': 'id', 0: 'full_text'})
    print(df.head())
    return df

def load_2021_and_remove_duplicates():
    df_2021 = load_2021()
    print(f'2021 shape: {df_2021.shape}')
    df_fb3 = pd.read_csv('../input/feedback-prize-english-language-learning/train.csv')
    id_2021 = df_2021['id'].values.tolist()
    id_fb3 = df_fb3['text_id'].values.tolist()
    duplicates = set(id_2021) & set(id_fb3)
    print(f'Found {len(duplicates)} duplicates')
    df_2021 = df_2021[~df_2021['id'].isin(duplicates)].reset_index(drop=True)
    print(f'2021 shape after removing duplicates: {df_2021.shape}')
    return df_2021



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
        wandb.init(project='mlm', entity='ktm98',
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

    if CFG.use_2021:
        train_df_2021 = load_2021()
        train_df = pd.concat([train_df, train_df_2021], axis=0)

    train_df = process(train_df)

    if debug:
        train_df = train_df.head(100)

    train_text_list, val_text_list = train_test_split(train_df['full_text'].values, test_size=0.25, random_state=CFG.seed)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)

    if CFG.add_new_token:
        new_tokens = ['STUDENT_NAME', 'PROPER_NAME', 'Generic_Name', 'Generic_School', 'LOCATION_NAME', 'OTHER_NAME', 'RESTAURANT_NAME',
                    'TEACHER_NAME', 'STORE_NAME', 'LANGUAGE_NAME', 'Generic_City', '']
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})


    tokenizer.save_pretrained(CFG.output_dir+'tokenizer/')
    CFG.tokenizer = tokenizer

    
    # =============== mlm ===============
    mlm_train_json_path = CFG.output_dir + f'train_mlm.json'
    mlm_valid_json_path = CFG.output_dir + f'valid_mlm.json'

    for json_path, list_ in zip([mlm_train_json_path, mlm_valid_json_path],
                                [train_text_list, val_text_list]):
        with open(str(json_path), 'w') as f:
            for sentence in list_:
                row_json = {'text': sentence}
                json.dump(row_json, f)
                f.write('\n')
    datasets = load_dataset(
        'json',
        data_files={'train': str(mlm_train_json_path),
                    'valid': str(mlm_valid_json_path)},
        )

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=CFG.max_len)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        batch_size=CFG.batch_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=CFG.mlm_probability)
    config = AutoConfig.from_pretrained(CFG.model, output_hidden_states=True)
    model = AutoModelForMaskedLM.from_pretrained(CFG.model, config=config)


    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        evaluation_strategy="epoch",
        learning_rate=CFG.lr,
        weight_decay=CFG.weight_decay,
        save_strategy='no',
        per_device_train_batch_size=CFG.batch_size,
        num_train_epochs=CFG.epochs,
        # report_to="wandb",
        run_name=exp_name,
        logging_dir=CFG.output_dir + 'logs/',
        lr_scheduler_type=CFG.scheduler,
        warmup_ratio=CFG.warmup_ratio,
        fp16=True,
        logging_steps=500,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        seed=CFG.seed,
        gradient_checkpointing=CFG.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets['valid'],
        data_collator=data_collator,
        # optimizers=(optimizer, scheduler)
    )

    trainer.train()
    trainer.model.save_pretrained(CFG.output_dir + f'mlm_{CFG.model.split("/")[-1]}')


if __name__ == '__main__':
    main()