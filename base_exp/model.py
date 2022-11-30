import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, features):
        # ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(features)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(all_layer_embedding.size()).float()

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())

        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        # features.update({'token_embeddings': weighted_average})
        return weighted_average
    

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False, output_size=6):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            print(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.cfg.weighted_layer_pooling:
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=self.cfg.layer_start)
            if self.cfg.pooling == 'mean':
                self.mean_pool = MeanPooling()
            elif self.cfg.pooling == 'attention':
                self.attention_pool = AttentionPooling(self.config.hidden_size)
        else:
            if self.cfg.pooling == 'mean':
                self.pool = MeanPooling()
            elif self.cfg.pooling == 'attention':
                self.pool = AttentionPooling(self.config.hidden_size)
        if self.cfg.use_msd:
            self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(cfg.n_msd)])
        if self.cfg.concat_layers:
            self.fc = nn.Linear(self.config.hidden_size*4, output_size)
        else:
            self.fc = nn.Linear(self.config.hidden_size, output_size)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.cfg.init_method in ['normal_']:
                getattr(torch.nn.init, self.cfg.init_method)(module.weight, mean=0.0, std=self.config.initializer_range)
            elif 'orthogonal' in self.cfg.init_method:
                torch.nn.init.orthogonal_(module.weight.data)
            elif self.cfg.init_method in ['xavier_normal_' , 'xavier_uniform_', 'kaiming_normal_', 'kaiming_uniform_']:
                getattr(torch.nn.init, self.cfg.init_method)(module.weight)
            
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        if self.cfg.weighted_layer_pooling:
            feature = self.pool(outputs['hidden_states'])
            feature = self.mean_pool(feature, inputs['attention_mask'])
            # feature = features['token_embeddings'][:, 0]
        elif self.cfg.concat_layers:
            feature = torch.cat([self.pool(outputs['hidden_states'][-1-i], inputs['attention_mask']) for i in range(4)], dim=1)
        else:
            last_hidden_states = outputs[0]
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
    
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        if self.cfg.use_msd:
            output = sum([self.fc(dropout(feature)) for dropout in self.dropouts])/self.cfg.n_msd
        else:
            output = self.fc(feature)
        return output

    def freeze(self, freeze_layers=12):
        self.model.embeddings.requires_grad_(False)
        if 'funnel-transformer-xlarge' in self.cfg.model:
            self.model.encoder.block[:freeze_layers].requires_grad_(False)
        else:
            self.model.encoder.layer[:freeze_layers].requires_grad_(False)



def reinit_bert(model, reinit_layers):
    for layer in model.model.encoder.layer[-reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    return model