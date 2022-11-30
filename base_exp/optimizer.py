from torch.optim import Optimizer

class PriorWD(Optimizer):
    def __init__(self, optim, use_prior_wd=False, exclude_last_group=True):
        super(PriorWD, self).__init__(optim.param_groups, optim.defaults)
        self.param_groups = optim.param_groups
        self.optim = optim
        self.use_prior_wd = use_prior_wd
        self.exclude_last_group = exclude_last_group
        self.weight_decay_by_group = []
        for i, group in enumerate(self.param_groups):
            self.weight_decay_by_group.append(group["weight_decay"])
            group["weight_decay"] = 0

        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                self.prior_params[id(p)] = p.detach().clone()

    def step(self, closure=None):
        if self.use_prior_wd:
            for i, group in enumerate(self.param_groups):
                for p in group["params"]:
                    if self.exclude_last_group and i == len(self.param_groups):
                        p.data.add_(-group["lr"] * self.weight_decay_by_group[i], p.data)
                    else:
                        p.data.add_(
                            -group["lr"] * self.weight_decay_by_group[i], p.data - self.prior_params[id(p)],
                        )
        loss = self.optim.step(closure)

        return loss

    def compute_distance_to_prior(self, param):
        assert id(param) in self.prior_params, "parameter not in PriorWD optimizer"
        return (param.data - self.prior_params[id(param)]).pow(2).sum().sqrt()



def get_optimizer_grouped_parameters_1(
    model, model_type, 
    learning_rate, weight_decay, 
    layerwise_learning_rate_decay,
    decoder_lr
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    optimizer_grouped_parameters += [
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

def get_optimizer_grouped_parameters_2(model, learning_rate, weight_decay, decoder_lr):
    no_decay = ["bias", "LayerNorm.weight"]
    n_hidden = model.config.num_hidden_layers
    group_all = [f'layer.{i}.' for i in range(n_hidden)]
    group1= group_all[:n_hidden//3]
    group2= group_all[n_hidden//3:2*n_hidden//3]    
    group3= group_all[2*n_hidden//3:]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': weight_decay, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': weight_decay, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if 'model' not in n], 'lr':decoder_lr, "weight_decay": 0.0, },
    ]
    return optimizer_grouped_parameters