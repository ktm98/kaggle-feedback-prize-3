from cv2 import correctMatches
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import wandb
import sys
import os
import gc
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW
# from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pandas as pd

from tqdm import tqdm

sys.path.append('base_exp')
from utils import AverageMeter, timeSince, get_score
from dataset import collate, TrainDataset
from model import CustomModel, reinit_bert, QuantizedModel
from loss import RMSELoss
from mixout import replace_mixout
from optimizer import PriorWD, get_optimizer_grouped_parameters_1, get_optimizer_params, get_optimizer_grouped_parameters_2
from process import process
from adversarial import FGM, AWP

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, CFG):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    if epoch >= CFG.adv_start_epoch:
        if CFG.fgm:
            fgm = FGM(model)
        if CFG.awp:
            awp = AWP(model,
                    optimizer, 
                    adv_lr = CFG.adv_lr, 
                    adv_eps = CFG.adv_eps, 
                    scaler = scaler,
                    apex=CFG.apex,
                    criterion=criterion
                    )

    for step, (inputs, kd_labels, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        kd_labels = kd_labels.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels) + criterion(y_preds, kd_labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        if epoch >= CFG.adv_start_epoch:
            if CFG.awp:
                awp.attack_backward(inputs, labels, epoch)
            # Fast Gradient Method (FGM)
            if CFG.fgm:
                fgm.attack()
                with torch.cuda.amp.autocast(enabled = CFG.apex):
                    y_preds = model(inputs)
                    loss_adv = criterion(y_preds, labels)
                    loss_adv.backward()
                fgm.restore()
        if CFG.unscale:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)


        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if CFG.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device, CFG):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, kd_labels, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        kd_labels = kd_labels.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.inference_mode():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels) + criterion(y_preds, kd_labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, CFG, device='cuda:0'):

    print(f"========== fold: {fold} training ==========")

    target_cols = CFG.target_cols
    if CFG.use_unique_token_as_target:
        target_cols += ['n_unique_tokens']

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    if CFG.train_all:
        valid_folds = train_folds.head(1000)
    else:
        valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[target_cols].values
    valid_labels_original = valid_folds[CFG.target_cols].values

    if CFG.use_pl:
        print('using pl')
        # fb3_df = train_folds[train_folds['fold']>=0].reset_index(drop=True)
        pl_dfs = [pd.read_csv(os.path.join('../result', pl_exp_name, f'2021_pl_fold{fold}.csv')) for pl_exp_name in CFG.pl_exp_name]
        pseudo_targets = np.mean([df[CFG.target_cols].values for df in pl_dfs], axis=0)
        pl_df = pl_dfs[0]
        pl_df[CFG.target_cols] = pseudo_targets
        pl_df[[f'pred_{c}' for c in CFG.target_cols]] = pseudo_targets
        pl_df = process(pl_df)
        pl_df['n_unique_tokens'] = pl_df['full_text'].apply(lambda x: len(set(x.split()))//100)
        len_pl = len(pl_df)
        pl_df = pl_df.sample(frac=1, random_state=CFG.seed+fold).reset_index(drop=True)
        pl_dfs = []
        for epoch in range(CFG.epochs):
            start = int(CFG.pl_frac*len_pl*epoch)
            end = int(CFG.pl_frac*len_pl*(epoch+1))
            if end > len_pl and start < len_pl:
                end = end%len_pl
                pl_dfs.append(pd.concat([pl_df.iloc[start:], pl_df.iloc[:end]], axis=0).reset_index(drop=True))
            else:
                start = start % len_pl
                end = end % len_pl
                pl_dfs.append(pl_df.iloc[start:end].reset_index(drop=True))

            # print(pl_dfs[0].head(5))

        train_folds_ = pd.concat([train_folds, pl_dfs[0]], axis=0).reset_index(drop=True)
    else:
        train_folds_ = train_folds


    train_dataset = TrainDataset(CFG, train_folds_, max_len=CFG.max_len, target_cols=target_cols)
    valid_dataset = TrainDataset(CFG, valid_folds, max_len=CFG.max_len, target_cols=target_cols)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size ,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    # ====================================================
    # model & optimizer
    # ====================================================
    if CFG.quantize:
        model = QuantizedModel(CFG, config_path=None, pretrained=True, output_size=len(target_cols))

        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        import copy
        model_fused =copy.copy(model)
        model_fused.qconfig = quantization_config
        model_fused = torch.quantization.prepare_qat(model_fused)
        example_inputs = {'input_ids': torch.zeros((1, 10)).long(), 'attention_mask': torch.zeros((1, 10)).long(), 'token_type_ids': torch.zeros((1, 10)).long()}
        o1 = model(example_inputs)
        o2 = model_fused(example_inputs)
        print(o1.sum() - o2.sum())
    else:
        model = CustomModel(CFG, config_path=None, pretrained=True, output_size=len(target_cols))
    if os.path.exists(CFG.pretrained_path):
        model.model.from_pretrained(os.path.join(CFG.pretrained_path, f'mlm_{CFG.model.split("/")[-1]}'))
    #     model.model.load_state_dict(torch.load(os.path.join(CFG.pretrained_path, f'mlm_{CFG.model.split("/")[-1]}/pytorch_model.bin'), map_location='cpu'))
    if CFG.reinit_layers > 0:
        model = reinit_bert(model, CFG.reinit_layers)
    if CFG.use_mixout:
        model = replace_mixout(model, CFG.mixout_prob)
    if CFG.freeze_layers > 0:
        model.freeze(CFG.freeze_layers)
    torch.save(model.config, CFG.output_dir+'config.pth')
    model.to(device)

   
    if CFG.lr_method == 'constant':
        optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    elif CFG.lr_method == 'decay':
        optimizer_parameters = get_optimizer_grouped_parameters_1(
            model, 'model', learning_rate=CFG.encoder_lr, weight_decay=CFG.weight_decay,
            layerwise_learning_rate_decay=CFG.layerwise_lr_decay,
            decoder_lr=CFG.decoder_lr
        )
    elif CFG.lr_method == 'step':
        optimizer_parameters = get_optimizer_grouped_parameters_2(
            model, learning_rate=CFG.encoder_lr, weight_decay=CFG.weight_decay,
            decoder_lr=CFG.decoder_lr
        )
    
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr,
                      eps=CFG.eps, betas=CFG.betas)
    if CFG.use_prior_wd:
        optimizer = PriorWD(optimizer, use_prior_wd=CFG.use_prior_wd)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(cfg.warmup_ratio*num_train_steps), num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(cfg.warmup_ratio*num_train_steps), num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds_) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction='mean')  # RMSELoss(reduction="mean")

    best_score = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        if CFG.use_pl:
            train_loader.dataset.set_df(pd.concat([train_folds, pl_dfs[epoch]], axis=0).reset_index(drop=True))
        avg_loss = train_fn(fold, train_loader, model,
                            criterion, optimizer, epoch, scheduler, device, CFG)

        # eval
        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, device, CFG)

        # scoring
        if CFG.use_unique_token_as_target:
            predictions = predictions[:, :-1]
        score, scores = get_score(valid_labels_original, predictions)

        elapsed = time.time() - start_time

        print(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1,
                       f"[fold{fold}] avg_train_loss": avg_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})

        if best_score > score:
            best_score = score
            print(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       CFG.output_dir+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(CFG.output_dir+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds



# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions