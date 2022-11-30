import torch
from torch.utils.data import Dataset


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text, max_len=512):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df, max_len=512, target_cols=None):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.target_cols = target_cols
        self.labels = df[self.target_cols].values
        self.kd_labels = df[['pred_' + col for col in self.target_cols]].values
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item], max_len=self.max_len)
        kd_label = torch.tensor(self.kd_labels[item], dtype=torch.float)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, kd_label, label
    
    def set_df(self, df):
        self.texts = df['full_text'].values
        self.labels = df[self.target_cols].values
        self.kd_labels = df[['pred_' + col for col in self.target_cols]].values
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item], max_len=self.cfg.max_len)
        return inputs