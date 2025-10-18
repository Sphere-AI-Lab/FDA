import copy
import random
import re
import os
import torch
import torch.nn as nn
from utils.load_config import cache_dir
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import sys


def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    names = []
    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(model, names, params):
    for name, p in zip(names, params):
        set_attr(model, name.split("."), p)


def del_ex(model, exclude):
    new_model = copy.deepcopy(model)
    for param_name, param_value in model.named_parameters():
        exc = [re.match(regex, param_name) for regex in exclude]
        if any(exc):
            del_attr(new_model, param_name.split("."))
    return new_model

class MergedModel(nn.Module):
    def __init__(self, pretrained_model, models, init_scale):  
        super(MergedModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.models = models
        self.init_scale = init_scale

        # Freeze pretrained model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # Initialize alpha parameters for elementwise granularity
        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha = nn.ParameterList()
            for param in model.parameters():
                alpha.append(nn.Parameter(torch.ones_like(param) * self.init_scale, requires_grad=True))
            self.alphas.append(alpha)

        # Create a deep copy of the pretrained model for merging
        self.merged_model = copy.deepcopy(self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        merged_param = []
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                # Use alpha for elementwise granularity
                alpha = self.alphas[k][idx]
                param += alpha * (dict(self.models[k].named_parameters())[name] - pretrained_param)
            param += pretrained_param
            merged_param.append(param)

        # Assuming `load_weights` is a function that loads the merged parameters into the model
        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def forward(self, x):
        merged_model = self.get_merged_model()
        if hasattr(x, 'keys'):
            return merged_model(**x)
        else:
            return merged_model(x)



def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class LabeledDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_indices = []
        for i, dataset in enumerate(datasets):
            self.dataset_indices.extend(
                [(i, idx) for idx in range(len(dataset))])
        random.shuffle(self.dataset_indices)

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.dataset_indices[index]
        sample = self.datasets[dataset_idx][sample_idx]
        return sample, dataset_idx


def custom_collate_fn(batch):
    # Custom collate function to handle varying input sizes
    data = [item[0] for item in batch]
    source_loader = torch.tensor([item[1] for item in batch])
    return {'data': data, 'source_loader': source_loader}


def merge_data_loaders_from_trainers(trainers, batch_size=16, num_workers=0):
    # Extract datasets from the data loaders
    datasets = []
    for trainer in trainers:
        dataloader = trainer.get_train_dataloader()
        dataset = []
        for item in dataloader:
            dataset.append(trainer._prepare_inputs(item))
        datasets.append(dataset)

    # Create a merged dataset
    merged_dataset = LabeledDataset(datasets)

    # Create a new data loader from the merged dataset
    merged_loader = DataLoader(
        merged_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return merged_loader


class TransformedDataDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def transformed_data_collate_fn(batch):
    # data = torch.stack([item[0] for item in batch])
    # source_loaders = torch.tensor([item[1] for item in batch])
    # attention_mask = torch.stack([item[2] for item in batch])
    data = batch[0][0]
    source_loaders = batch[0][1]
    attention_mask = batch[0][2]
    return {'data': data, 'source_loader': source_loaders, 'attention_mask': attention_mask}


def transform_data_loader_prelayer(data_loader, model, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']

            # output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            # output[1] is attention mask, with shape [batch_size, 1, seq_length, seq_length]
            output = model(x)

            # batchsize = 1
            transformed_data.append(
                (output[0].cpu(), source_loader, output[1].cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_prelayer_pertask(data_loader, merged_model, models, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']

            inputs = []
            attention_masks = []

            # model_output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            # model_output[1] is attention mask, with shape [batch_size, 1, seq_length, seq_length]
            model_output = merged_model(x)
            inputs.append(model_output[0])
            attention_masks.append(model_output[1])
            # print(model_output[1])

            model = models[source_loader.item()]
            model_output = model(x)
            inputs.append(model_output[0])
            attention_masks.append(model_output[1])

            # shape of inputs: [2, batch_size, seq_length, embedding_dim] -> [batch_size, 2, seq_length, embedding_dim]
            # shape of attention_masks: [2, batch_size, 1, seq_length, seq_length] -> [batch_size, 2, 1, seq_length, seq_length]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()
            if attention_masks[0] is not None:
                attention_masks = torch.stack(
                    attention_masks).permute(1, 0, 2, 3, 4).cpu()
            else:
                print('here')# 基本上就不使用此处。表明，使用的attention mask就是原先构造数据集时得到的
                attention_masks = torch.zeros(
                    inputs.shape[0], inputs.shape[1], 1, inputs.shape[2], inputs.shape[2])

            transformed_data.append((inputs, source_loader, attention_masks))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_layer_pertask(data_loader, merged_model, models, device, num_workers=1, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            # shape of x: [batch_size, 2, seq_length, embedding_dim] -> [2, batch_size, seq_length, embedding_dim]
            x = data['data'].to(device)
            x = x.permute(1, 0, 2, 3)

            source_loader = data['source_loader']

            # shape of attention mask: [batch_size, 2, 1, seq_length, seq_length] -> [2, batch_size, 1, seq_length, seq_length]
            attention_mask = data['attention_mask'].to(device)
            attention_mask = attention_mask.permute(1, 0, 2, 3, 4)

            inputs = []

            output = merged_model(
                x[0], attention_mask[0], None, None, None, None, False)[0]
            inputs.append(output)

            model = models[source_loader.item()]
            output = model(x[1], attention_mask[1], None,
                           None, None, None, False)[0]
            inputs.append(output)

            # shape of inputs: [2, batch_size, seq_length, embedding_dim] -> [batch_size, 2, seq_length, embedding_dim]
            # shape of attention_masks: [2, batch_size, 1, seq_length, seq_length] -> [batch_size, 2, 1, seq_length, seq_length]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()
            attention_mask = attention_mask.permute(1, 0, 2, 3, 4).cpu()

            # batchsize = 1
            transformed_data.append(
                (inputs, source_loader, attention_mask.cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def load_pretrained_model(args):
    try:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=os.path.join(args.model_location, args.model)).to(args.device)
    except:
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=args.model, cache_dir=cache_dir).to(args.device)

    return pretrained_model


def load_fine_tuned_model(args, dataset_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.load_model_paths_dict[dataset_name]).to(args.device)
    return model



def load_avg_merged_model(args, merge_coef=0.3):
    pretrained_model = load_pretrained_model(args).cpu()
    new_state_dict = pretrained_model.state_dict()
    merge_coef=args.alpha
    with torch.no_grad():
        for dataset in args.dataset_names:
            model = load_fine_tuned_model(args, dataset).cpu()

            model_params = model.state_dict()
            pretrained_params = pretrained_model.state_dict()

            for name in pretrained_params:
                if 'classifier' not in name:
                    delta = (model_params[name] - pretrained_params[name]) * merge_coef
                    new_state_dict[name] += delta

            # 释放当前模型
            del model, model_params
            torch.cuda.empty_cache()

    pretrained_model.load_state_dict(new_state_dict)

    if args.init_params=='tsv':
        merged_model=torch.load(f'/data/shikexuan/nlu_model/{args.model}/merged_tsv.pt').state_dict()
        pretrained_model.load_state_dict(merged_model)
        del merged_model
    elif args.init_params=='wudi':
        merged_model=torch.load(f'/data/shikexuan/nlu_model/{args.model}/merged_wudi.pt').state_dict()
        pretrained_model.load_state_dict(merged_model)
        del merged_model

    # 最后再放到 GPU
    pretrained_model = pretrained_model.to(args.device)

    return pretrained_model


def load_merged_layers(args, layer_idx):
    pretrained_model = load_pretrained_model(args)

    if args.model == 'roberta-base':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]
    elif args.model == 'roberta-large':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]

    if args.init_params=='ta':
        layers = []
        for dataset in args.dataset_names:
            model = load_fine_tuned_model(args, dataset)
            if args.model == 'roberta-base':
                layer = model.roberta.encoder.layer[layer_idx]
            elif args.model == 'roberta-large':
                layer = model.roberta.encoder.layer[layer_idx]
            layers.append(layer)
            del model
            torch.cuda.empty_cache()
        merged_layers = MergedModel(layer_pretrained, layers ,init_scale=args.alpha)
        del pretrained_model
        torch.cuda.empty_cache()
        return merged_layers
    elif args.init_params=='tsv':
        merged_encoder = torch.load(f'/data/shikexuan/nlu_model/{args.model}/merged_tsv.pt',map_location=args.device)
        merged_layer=merged_encoder.roberta.encoder.layer[layer_idx]
        layer_merged=[]
        layer_merged.append(merged_layer)
        del merged_encoder
        torch.cuda.empty_cache()
        merged_layers = MergedModel(layer_pretrained, layer_merged,init_scale=1.0)
        return merged_layers
    elif args.init_params=='wudi':
        merged_encoder = torch.load(f'/data/shikexuan/nlu_model/{args.model}/merged_wudi.pt',map_location=args.device)
        merged_layer=merged_encoder.roberta.encoder.layer[layer_idx]
        layer_merged=[]
        layer_merged.append(merged_layer)
        del merged_encoder
        torch.cuda.empty_cache()
        merged_layers = MergedModel(layer_pretrained, layer_merged,init_scale=1.0)
        return merged_layers



def load_pretrained_layer(args, layer_idx):
    pretrained_model = load_pretrained_model(args)

    if args.model == 'bert-base-uncased':
        layer_pretrained = pretrained_model.bert.encoder.layer[layer_idx]
    elif args.model == 'roberta-base':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]
    elif args.model == 'roberta-large':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]

    return layer_pretrained



def new_forward_bert(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                     head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                     output_hidden_states=None, return_dict=None):
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]
    # pooled_output.shape = [batchsize, 768]
    return pooled_output


def new_forward_roberta(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                        head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                        output_hidden_states=None, return_dict=None):
    outputs = self.roberta(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]
    return sequence_output




def initialize_X(args, params):
    device = args.device
    attention_mask = None
    if args.model == 'roberta-base':
        feature_dim = 768
    elif args.model == 'roberta-large':
        feature_dim = 1024

    if 'random' in args.name:
        if args.model == 'roberta-base':
            X = torch.randn(args.anchor_num, args.token_num, 768, device=device)
        elif args.model == 'roberta-large':
            X = torch.randn(args.anchor_num,args.token_num, 1024, device=device)
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        X=X*args.scale
        X.requires_grad_()
    else:
        if args.model == 'roberta-base':
            feature_dim = 768
        elif args.model == 'roberta-large':
            feature_dim = 1024
        all_features = []
        param_features=[]
        for p in params:
            if p.ndim == 2 and p.shape[1] == feature_dim:
                param_features.append(p.detach().reshape(-1, feature_dim))
        param_features=torch.cat(param_features,dim=0)

        all_features =param_features
        
        assert all_features.shape[0] >= args.token_num * args.anchor_num
        idx = torch.randperm(all_features.shape[0])[:args.token_num * args.anchor_num]
        X = all_features[idx].reshape(args.anchor_num,args.token_num, feature_dim).to(device)
        X.requires_grad_()  
        
    return X,attention_mask

class LabeledMultiDatasetFromDict(Dataset):
    def __init__(self, X_dict):
        """
        X_dict: {
            'dataset_name': {
                'input': Tensor [batch_size, seq_len, hidden_dim],
                'gt': Tensor [batch_size, seq_len, hidden_dim],
                'attention_mask': Tensor [batch_size, 1, seq_len, seq_len]
            }, ...
        }
        """
        self.inputs = []
        self.labels = []
        self.masks = []
        for name, pair in X_dict.items():
            X = pair["input"]           # [B, L, D]
            Y = pair["gt"]              # [B, L, D]
            mask = pair["attention_mask"]  # [B, 1, L, L]
            self.inputs.extend(X)
            self.labels.extend(Y)
            self.masks.extend(mask) # extend就是逐元素加入

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.masks[idx]
