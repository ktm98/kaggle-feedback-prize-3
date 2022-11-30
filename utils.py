import argparse
import collections
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from operator import is_

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoost, CatBoostRegressor
from datatable import first
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, mean_squared_error
from tqdm.auto import tqdm
import torch
import pathlib
import joblib


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_args():
    # 引数の導入
    parser = argparse.ArgumentParser(description='YAMLありの例')
    parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')  # 
    parser.add_argument('--debug', action='store_true', help='引数が指定されたらTrue、指定されなかったらFalse')
    parser.add_argument('--tuning', action='store_true', help='引数が指定されたらTrue、指定されなかったらFalse')
    args = parser.parse_args()
    return args

# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_env_notebook():
    """Determine wheather is the environment Jupyter Notebook"""
    # if 'get_ipython' not in globals():
    #     # Python shell
    #     return False
    # env_name = get_ipython().__class__.__name__
    # if env_name == 'TerminalInteractiveShell':
    #     # IPython shell
    #     return False
    # # Jupyter Notebook
    # return True
    return hasattr(__builtins__,'__IPYTHON__')
            
def show_importances(models, save_path=None):
    # 特徴量重要度を保管する dataframe を用意
    feature_importances = pd.DataFrame()

    for fold, model in enumerate(models):
        # if model_name == 'lightgbm':
        #     model = lgb.Booster(model_file=path)
        # elif model_name == 'xgboost':
        #     model = xgb.Booster(model_file=path)
        # elif model_name == 'catboost':
        #     model = CatBoost().load_model(path)

        tmp = pd.DataFrame()
        if isinstance(model, xgb.Booster) :
            # tmp['feature'] = model.feature_names
            importance = model.get_score(importance_type='gain')
            tmp['feature'] = list(importance.keys())
            tmp['importance'] = list(importance.values())
        elif isinstance(model, CatBoost):
            
            tmp['feature'] = model.feature_names_
            tmp['importance'] = model.feature_importances_
        else:
            tmp['feature'] = model.feature_name()
            tmp['importance'] = model.feature_importance(importance_type='gain')
        tmp['fold'] = fold

        feature_importances = feature_importances.append(tmp)

    # 各特徴量で集約して、重要度の平均を算出。上位50個だけ抜き出す
    importances = (feature_importances.groupby("feature")["importance"].sum() / feature_importances.groupby("feature")["importance"].size()).sort_values(ascending=False)
    order = list(importances.index)[:50]
    print('important features highest 50')
    print(order)

    # 可視化
    fig = plt.figure(figsize=(10, 20))
    fig.patch.set_facecolor('white')
    sns.barplot(x='importance', y='feature', data=feature_importances, order=order)
    plt.title('LGBM importance highest 50')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path+'highest50.png')

    # if is_env_notebook():
    plt.show()
    
    # 下位50個
    order = order = list(importances.index)[-50:]
    print('important features lowest 50')
    print(order)

    # 可視化
    fig = plt.figure(figsize=(10, 20))
    plt.figure(figsize=(10, 20))
    fig.patch.set_facecolor('white')
    sns.barplot(x='importance', y='feature', data=feature_importances, order=order)
    plt.title('LGBM importance lowest 50')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path+'lowest50.png')
    # if is_env_notebook():
    plt.show()
    importances.to_csv(save_path + 'importance.csv')
    return importances
    
 
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.2f} s')

def fast_merge(df1, df2, on=None):
    return pd.concat([
            df1,
            df2.set_index(on)
                .reindex(df1[on].values)
                .reset_index(drop=True)
            ], axis=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def convert_dot_dict(dic: dict) -> dict:
    '''
    yamlで
    A.B: 0
    をロードした時に
    {'A,B': 0}
    となるのを
    {'A': {'B': 0}}
    とする
    '''
    def deep_dict():
        return defaultdict(deep_dict)

    result = deep_dict()

    def deep_insert(key, value):
        d = result
        keys = key.split(".")
        for subkey in keys[:-1]:
            d = d[subkey]
        d[keys[-1]] = value
        
    for k, v in dic.items():
        deep_insert(k, v)
        
    json_str = json.dumps(result, indent=4)
    return json.loads(json_str)


def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores