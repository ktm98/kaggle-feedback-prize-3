import numpy as np
import pandas as pd

import os
import sys
import argparse
import yaml

from utils import get_score, get_args

def ensemble(models, weights=None, verbose=True):
    if verbose:
        print('==============================================')
        print('Ensemble')
        print('==============================================')
        print('Models:', models)
        print('==============================================')
    if weights is not None:
        if verbose:
            print('Weights:', weights)
            print('==============================================')

    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    paths = [os.path.join('/root/result/', model, 'oof_df.csv') for model in models]
    dfs = [pd.read_csv(path).sort_values('text_id') for path in paths]
    for df, model_name in zip(dfs, models):
        score, scores = get_score(df[target_cols].values, df[[f"pred_{c}" for c in target_cols]].values)
        if verbose: print(f'{model_name}: score: {score:<.4f}  Scores: {scores}')
        # print(f"Score: {score:<.4f}  Scores: {scores}")
    if verbose: print('=======================')
    
    targets = dfs[0][target_cols].values
    ens_pred = np.zeros((len(dfs[0]), len(target_cols)))
    for i, target_col in enumerate(target_cols):
        preds = np.array([df[f"pred_{target_col}"].values for df in dfs])
        ens_pred[:, i] = np.average(preds, axis=0, weights=weights)
        if verbose:
            print(f'\n===== {target_col} Correlation =====')
            print(pd.DataFrame(preds.T, columns=models).corr())
            print('')
    
    score, scores = get_score(targets, ens_pred)
    if verbose:
        print('--------- ensemble score ----------')
        print(f'Score: {score:<.4f}  Scores: {scores}\n')

    dfs[0][[f'pred_{c}' for c in target_cols]] = ens_pred

    return dfs[0]


def main():
    args = get_args()
    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg['weights'] is not None:
        cfg['weights'] = np.array(cfg['weights']) / np.sum(cfg['weights'])
    ensemble(cfg['models'], cfg['weights'])

if __name__ == '__main__':
    main()