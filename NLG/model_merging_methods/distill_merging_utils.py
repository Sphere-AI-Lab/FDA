import copy
import random
import re
import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader


def initialize_X(args,pretrained_params,finetuned_params):
    if 'random' in args.name:
        X=torch.randn(args.anchor_num,5120).cuda()
        X=X*args.scale
    else:
        param_features=[]
        for p in finetuned_params:
            if p.ndim == 2 and p.shape[1] ==5120 :
                param_features.append(p.detach().reshape(-1, 5120))
        all_features=torch.cat(param_features,dim=0)
        if all_features.shape[0] >= args.anchor_num:
            idx = torch.randperm(all_features.shape[0])[:args.anchor_num]
            X = all_features[idx].cuda()
        else:
            needed = args.anchor_num - all_features.shape[0]
            existing = all_features
            extra = []
            for _ in range(needed):
                i, j = torch.randint(0, all_features.shape[0], (2,))
                mean_vec = (all_features[i] + all_features[j]) / 2
                extra.append(mean_vec.unsqueeze(0))
            extra = torch.cat(extra, dim=0)  # [needed, feature_dim]
            all_padded = torch.cat([existing, extra], dim=0)
            X = all_padded[:args.anchor_num].cuda()   # [anchor_num, feature_dim]
    X.requires_grad_()
    return X

class LabeledMultiDatasetFromDict(Dataset):
    def __init__(self, X_dict):
        self.inputs = []
        self.labels = []
        for name, pair in X_dict.items():
            X = pair["input"]   # [batch_size, seq_len, hidden_dim]
            Y = pair["gt"]  # [batch_size, seq_len, hidden_dim]
            self.inputs.extend(X)
            self.labels.extend(Y)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def check_gpu():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Cached memory (reserved): {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
        print()


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
    def __init__(self, pretrained_model, models, granularity):
        super(MergedModel, self).__init__()
        self.pretrained_model = pretrained_model
        # self.models = copy.deepcopy(models)
        self.models = models
        self.granularity = granularity

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha = nn.ParameterList()
            if self.granularity == 'taskwise':
                alpha.append(nn.Parameter(
                    torch.tensor(0.5), requires_grad=True))
            elif self.granularity == 'layerwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(
                        torch.tensor(0.5), requires_grad=True))
            elif self.granularity == 'elementwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.ones_like(
                        param) * 0.5, requires_grad=True))
            else:
                raise NotImplementedError(
                    f'Invalid granularity: {self.granularity}')
            self.alphas.append(alpha)

        self.merged_model = copy.deepcopy(
            self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        merged_param = []
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param.append(param)

        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def get_named_parameters(self):
        merged_param = {}
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param[name] = param
        return merged_param

    def forward(self, x):
        merged_model = self.get_merged_model()
        if isinstance(x, dict):
            return merged_model(**x)
        else:
            return merged_model(x)

    def turn_on_layer(self, layer_idx):
        layer_name = f'layer.{layer_idx}'
        assert self.granularity in ['layerwise', 'elementwise']
        for idx, (name, _) in enumerate(self.pretrained_model.named_parameters()):
            for k in range(len(self.models)):
                alpha = self.alphas[k][idx]
                if layer_name in name:
                    alpha.requires_grad = True
                else:
                    alpha.requires_grad = False

class MergedModel_FDAs(nn.Module):
    def __init__(self, pretrained_model, models, granularity,scale=0.5):
        super(MergedModel_FDAs, self).__init__()
        self.pretrained_model = pretrained_model
        self.models = models
        self.granularity = granularity

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha = nn.ParameterList()
            if self.granularity == 'taskwise':
                alpha.append(nn.Parameter(
                    torch.tensor(scale), requires_grad=True))
            elif self.granularity == 'layerwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(
                        torch.tensor(scale), requires_grad=True))
            elif self.granularity == 'elementwise':
                for param in model.parameters():
                    alpha.append(nn.Parameter(torch.ones_like(
                        param) * scale, requires_grad=True))
            else:
                raise NotImplementedError(
                    f'Invalid granularity: {self.granularity}')
            self.alphas.append(alpha)

        self.merged_model = copy.deepcopy(
            self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        merged_param = []
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param.append(param)

        load_weights(self.merged_model, self.names, merged_param)

        return self.merged_model

    def get_named_parameters(self):
        merged_param = {}
        for idx, (name, pretrained_param) in enumerate(self.pretrained_model.named_parameters()):
            param = torch.zeros_like(pretrained_param)
            for k in range(len(self.models)):
                if self.granularity == 'taskwise':
                    alpha = self.alphas[k][0]
                else:
                    alpha = self.alphas[k][idx]
                param += alpha * \
                    (dict(self.models[k].named_parameters())[
                        name] - pretrained_param)
            param += pretrained_param
            merged_param[name] = param
        return merged_param

    def forward(self, x):
        merged_model = self.get_merged_model()
        if isinstance(x, dict):
            return merged_model(**x)
        else:
            return merged_model(x)

    def turn_on_layer(self, layer_idx):
        layer_name = f'layer.{layer_idx}'
        assert self.granularity in ['layerwise', 'elementwise']
        for idx, (name, _) in enumerate(self.pretrained_model.named_parameters()):
            for k in range(len(self.models)):
                alpha = self.alphas[k][idx]
                if layer_name in name:
                    alpha.requires_grad = True
                else:
                    alpha.requires_grad = False


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
    if len(batch[0]) > 2:
        attention_mask = batch[0][2]
    else:
        return {'data': data, 'source_loader': source_loaders}
    return {'data': data, 'source_loader': source_loaders, 'attention_mask': attention_mask}


def transform_data_loader_prelayer(data_loader, model, device, num_workers=0, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            source_loader = data['source_loader']

            # output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            # output[1] is attention mask, with shape [batch_size, 1, seq_length, seq_length]
            output = model(x)

            # batchsize = 1
            transformed_data.append((output[0].cpu(), source_loader, output[1].cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_prelayer_pertask(data_loader, merged_model, models, device, num_workers=0, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            # print(x['input_ids'].shape)
            source_loader = data['source_loader']

            inputs = []
            attention_masks = []

            # model_output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            # model_output[1] is attention mask, with shape [batch_size, 1, seq_length, seq_length]
            model_output = merged_model(x)
            # print(model_output)
            # print(model_output[0].shape)
            inputs.append(model_output[0])
            if len(model_output) > 1:
                attention_masks.append(model_output[1])
            for model in models:
                model_output = model(x)
                inputs.append(model_output[0])
                if len(model_output) > 1:
                    attention_masks.append(model_output[1])

            # shape of inputs: [num_tasks+1, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks+1, seq_length, embedding_dim]
            # shape of attention_masks: [num_tasks+1, batch_size, 1, seq_length, seq_length] -> [batch_size, num_tasks+1, 1, seq_length, seq_length]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()
            if len(attention_masks) > 0:
                attention_masks = torch.stack(attention_masks).permute(1, 0, 2, 3, 4).cpu()

            # batchsize = 1
            if len(attention_masks) > 0:
                transformed_data.append((inputs, source_loader, attention_masks))
            else:
                transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_layer(data_loader, model, device, num_workers=0, shuffle=True, language_model_name='Llama-2-13b-hf_32001'):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'].to(device)
            source_loader = data['source_loader']
            if 'attention_mask' in data:
                attention_mask = data['attention_mask'].to(device)
            else:
                attention_mask = None
            if language_model_name == 'Llama-2-13b-hf_32001':
                output = model(x, attention_mask)[0]
            elif language_model_name == 'Qwen2.5-7B':
                output = model(x, attention_mask)[0]
            else:
                output = model(x, attention_mask, None, None, None, None, False)[0]

            # batchsize = 1
            if attention_mask is not None:
                transformed_data.append((output.cpu(), source_loader, attention_mask.cpu()))
            else:
                transformed_data.append((output.cpu(), source_loader))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def transform_data_loader_layer_pertask(data_loader, merged_model, models, device, num_workers=0, shuffle=True):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            # shape of x: [batch_size, num_tasks+1, seq_length, embedding_dim] -> [num_tasks+1, batch_size, seq_length, embedding_dim]
            x = data['data'].to(device)
            x = x.permute(1, 0, 2, 3)

            source_loader = data['source_loader']

            # shape of attention mask: [batch_size, num_tasks+1, 1, seq_length, seq_length] -> [num_tasks+1, batch_size, 1, seq_length, seq_length]
            attention_mask = data['attention_mask'].to(device)
            attention_mask = attention_mask.permute(1, 0, 2, 3, 4)

            inputs = []

            output = merged_model(x[0], attention_mask[0], None, None, None, None, False)[0]
            inputs.append(output)
            for idx, model in enumerate(models):
                output = model(x[idx+1], attention_mask[idx+1], None, None, None, None, False)[0]
                inputs.append(output)

            # shape of inputs: [num_tasks, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks, seq_length, embedding_dim]
            # shape of attention_masks: [num_tasks, batch_size, 1, seq_length, seq_length] -> [batch_size, num_tasks, 1, seq_length, seq_length]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()
            attention_mask = attention_mask.permute(1, 0, 2, 3, 4).cpu()

            # batchsize = 1
            transformed_data.append((inputs, source_loader, attention_mask.cpu()))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader


def create_pre_encoder_activation_extractor(model, language_model_name):
    def pre_encoder_activation_extractor(x):
        class StopForwardPass(Exception):
            pass

        encoder_input = []

        def create_hook():
            def hook(module, input, output):
                # wrong here
                encoder_input.append(input)
                raise StopForwardPass
            return hook

        if language_model_name == 'bert-base-uncased':
            block = model.bert.encoder.layer[0]
        elif language_model_name == 'roberta-base':
            block = model.roberta.encoder.layer[0]
        handle = block.register_forward_hook(create_hook())

        try:
            model(**x)
        except StopForwardPass:
            pass

        handle.remove()

        return encoder_input[0]

    return pre_encoder_activation_extractor


def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def load_pretrained_model(args):
    if args.language_model_name == 'Llama-2-13b-hf_32001' or args.language_model_name == 'Qwen2.5-7B':
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.path.join(args.cache_dir, args.language_model_name), args.llm_version), device_map=args.device)
        remove_grad(pretrained_model)
    else:
        try:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=os.path.join(args.cache_dir, args.language_model_name)).to(args.device)
        except:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=args.language_model_name, cache_dir=args.cache_dir).to(args.device)

    return pretrained_model

def load_pretrained_model_cpu(args):
    if args.language_model_name in ['Llama-2-13b-hf_32001', 'Qwen2.5-7B']:
        # 先在 CPU 上加载大模型
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(
                os.path.join(args.cache_dir, args.language_model_name),
                args.llm_version
            ),
            device_map="cpu"   # 强制加载到 CPU
        )
        remove_grad(pretrained_model)
    else:
        try:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=os.path.join(
                    args.cache_dir,
                    args.language_model_name
                ),
                device_map="cpu"   # 同样在 CPU 上加载
            )
        except:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=args.language_model_name,
                cache_dir=args.cache_dir,
                device_map="cpu"
            )

    return pretrained_model


def load_fine_tuned_model(args, dataset_name):
    if args.language_model_name == 'Llama-2-13b-hf_32001' or args.language_model_name == 'Qwen2.5-7B':
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(args.cache_dir, args.task_model_mapping_dict[dataset_name]), device_map=args.device)
        remove_grad(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=args.load_model_paths_dict[dataset_name]).to(args.device)
    return model

# This function also merged the classification head, which should be reloaded for each dataset
def load_avg_merged_model(args, merge_coef=0.3):
    pretrained_model = load_pretrained_model(args)

    new_state_dict = pretrained_model.state_dict()

    for dataset in args.dataset_names:
        model = load_fine_tuned_model(args, dataset)
        for name, param in pretrained_model.named_parameters():
            if 'classifier' not in name:
                new_param = (dict(model.named_parameters())[name]-dict(pretrained_model.named_parameters())[name]) * merge_coef
                new_state_dict[name] = new_state_dict[name] + new_param
        del model
        torch.cuda.empty_cache()

    pretrained_model.load_state_dict(new_state_dict)
    return pretrained_model


def load_merged_layers(args, layer_idx):
    pretrained_model = load_pretrained_model(args)

    if args.language_model_name == 'bert-base-uncased':
        layer_pretrained = pretrained_model.bert.encoder.layer[layer_idx]
    elif args.language_model_name == 'roberta-base':
        layer_pretrained = pretrained_model.roberta.encoder.layer[layer_idx]
    elif args.language_model_name == 'Llama-2-13b-hf_32001':
        layer_pretrained = pretrained_model.model.encoder.layers[layer_idx]
    elif args.language_model_name == 'Qwen2.5-7B':
        layer_pretrained = pretrained_model.model.encoder.layers[layer_idx]

    layers = []
    for dataset in args.dataset_names:
        model = load_fine_tuned_model(args, dataset)
        if args.language_model_name == 'bert-base-uncased':
            layer = model.bert.encoder.layer[layer_idx]
        elif args.language_model_name == 'roberta-base':
            layer = model.roberta.encoder.layer[layer_idx]
        elif args.language_model_name == 'Llama-2-13b-hf_32001':
            layer = model.model.encoder.layers[layer_idx]
        elif args.language_model_name == 'Qwen2.5-7B':
            layer = model.model.encoder.layers[layer_idx]
        layers.append(layer)

    merged_layers = MergedModel(layer_pretrained, layers, 'elementwise')

    return merged_layers, layers


def get_weight_map_llm(model_name, args):
    # model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'{args.llm_version}/split')
    model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'split')
    weight_map = json.load(open(os.path.join(model_path, 'model_index.json')))
    return weight_map
# def get_weight_map_llm(model_name, args):
#     model_path = os.path.join(os.path.join(args.cache_dir, model_name))
#     weight_map = json.load(open(os.path.join(model_path, 'split/model_index.json')))
#     return weight_map

def load_part_model(args, module_name, model_name):
    weight_map = get_weight_map_llm(model_name, args)
    # model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'{args.llm_version}/split')
    model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'split')
    weight_path = os.path.join(model_path, weight_map[module_name])
    model = torch.load(weight_path).to(args.device)
    remove_grad(model)
    return model

def get_weight_map_llm_for_merged(args,path):
    model_path = os.path.join(path, f'split')
    weight_map = json.load(open(os.path.join(model_path, 'model_index.json')))

    return weight_map

def load_part_model_for_merged(args, path,module_name):
    weight_map = get_weight_map_llm_for_merged(args,path)
    model_path = os.path.join(path, f'split')
    weight_path = os.path.join(model_path, weight_map[module_name])
    model = torch.load(weight_path).to(args.device)
    remove_grad(model)
    return model


def load_merged_layers_llm(args, layer_idx):
    layer_pretrained = load_part_model(args, f'model.layers.{layer_idx}', args.language_model_name)

    layers = []
    for dataset in args.dataset_names:
        layer = load_part_model(args, f'model.layers.{layer_idx}', args.task_model_mapping_dict[dataset])
        layers.append(layer)

    merged_layers = MergedModel(layer_pretrained, layers, 'elementwise')

    return merged_layers, layers

def load_avg_merged_model_llm_cpu(args, merge_coef=0.5):
    pre_model = load_pretrained_model_cpu(args)

    modules = ['model.embed_tokens.', 'model.norm.', 'lm_head.']
    for i in range(40):
        modules.append(f'model.layers.{i}.')

    for mod in modules:
        for name, param in pre_model.named_parameters():
            # flag = False
            if mod not in name:
                continue
            value = dict(pre_model.named_parameters())[name].clone()
            for dataset in args.dataset_names:
                model = load_part_model(args, mod[:-1], args.task_model_mapping_dict[dataset])
                model=model.cpu()
                # print(model)
                # for _name, _param in model.named_parameters():
                #     if _name == name[len(mod):]:
                #         print(_name, name)
                #         flag = True
                    # print(_name)
                value += (dict(model.named_parameters())[name[len(mod):]] - dict(pre_model.named_parameters())[name]) * merge_coef
                del model
                torch.cuda.empty_cache()
            # print(torch.norm(dict(pre_model.named_parameters())[name]))
            # print(torch.norm(value))
            # print(torch.norm(dict(pre_model.named_parameters())[name] - value))
            set_attr(pre_model, name.split('.'), nn.Parameter(value, requires_grad=False))
            # print(torch.norm(dict(pre_model.named_parameters())[name]))
            # print(torch.norm(value))
            # print(torch.norm(dict(pre_model.named_parameters())[name] - value))
            del value
            # if not flag:
            #     print(name)
            #     raise ValueError

    return pre_model

def load_avg_merged_model_llm(args, merge_coef=0.5):
    pre_model = load_pretrained_model(args)

    modules = ['model.embed_tokens.', 'model.norm.', 'lm_head.']
    for i in range(40):
        modules.append(f'model.layers.{i}.')

    for mod in modules:
        for name, param in pre_model.named_parameters():
            # flag = False
            if mod not in name:
                continue
            value = dict(pre_model.named_parameters())[name].clone()
            for dataset in args.dataset_names:
                model = load_part_model(args, mod[:-1], args.task_model_mapping_dict[dataset])
                # print(model)
                # for _name, _param in model.named_parameters():
                #     if _name == name[len(mod):]:
                #         print(_name, name)
                #         flag = True
                    # print(_name)
                value += (dict(model.named_parameters())[name[len(mod):]] - dict(pre_model.named_parameters())[name]) * merge_coef
                del model
                torch.cuda.empty_cache()
            # print(torch.norm(dict(pre_model.named_parameters())[name]))
            # print(torch.norm(value))
            # print(torch.norm(dict(pre_model.named_parameters())[name] - value))
            set_attr(pre_model, name.split('.'), nn.Parameter(value, requires_grad=False))
            # print(torch.norm(dict(pre_model.named_parameters())[name]))
            # print(torch.norm(value))
            # print(torch.norm(dict(pre_model.named_parameters())[name] - value))
            del value
            # if not flag:
            #     print(name)
            #     raise ValueError

    return pre_model


def load_avg_merged_model_pre_llm(args, merge_coef=0.5):
    pre_model = load_pretrained_model(args).model
    check_gpu()
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    for dataset in args.dataset_names:
        model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset])
        for name, param in model.named_parameters():
            new_param = (dict(model.named_parameters())[name] - dict(pre_model.named_parameters())[f'embed_tokens.{name}']) * merge_coef
            if new_state_dict.get(f'embed_tokens.{name}') is None:
                new_state_dict[f'embed_tokens.{name}'] = new_param
            else:
                new_state_dict[f'embed_tokens.{name}'] += new_param
        del model
        torch.cuda.empty_cache()

    for name, value in new_state_dict.items():
        set_attr(pre_model, name.split('.'), nn.Parameter(value + dict(pre_model.named_parameters())[name], requires_grad=False))

    return pre_model

def load_avg_merged_model_pre_llm_cpu(args, merge_coef=0.5):
    # 强制加载到 CPU
    pre_model = load_pretrained_model_cpu(args).model
    check_gpu()

    # 删除不需要的部分，释放内存
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    pre_params = dict(pre_model.named_parameters())

    for dataset in args.dataset_names:
        # 仍然在 CPU 上加载
        model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset]).to("cpu")
        model_params = dict(model.named_parameters())

        for name, param in model_params.items():
            new_param = (param - pre_params[f'embed_tokens.{name}']) * merge_coef
            if f'embed_tokens.{name}' not in new_state_dict:
                new_state_dict[f'embed_tokens.{name}'] = new_param
            else:
                new_state_dict[f'embed_tokens.{name}'] += new_param

        del model
        torch.cuda.empty_cache()

    # 用差分更新 pre_model 的参数
    for name, value in new_state_dict.items():
        set_attr(
            pre_model,
            name.split('.'),
            nn.Parameter(value + pre_params[name], requires_grad=False)
        )

    # 保留在 CPU，不占用显存
    return pre_model.to("cpu")

def load_single_merged_model_pre_llm_cpu(args, dataset):
    # 强制加载到 CPU
    pre_model = load_pretrained_model_cpu(args).model
    check_gpu()

    # 删除不需要的部分，减少内存
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    # 同样在 CPU 上加载部分模型
    model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset]).to("cpu")

    # 参数差分：全部在 CPU 上计算
    pre_params = dict(pre_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name, param in model_params.items():
        new_param = param - pre_params[f'embed_tokens.{name}']
        if f'embed_tokens.{name}' not in new_state_dict:
            new_state_dict[f'embed_tokens.{name}'] = new_param
        else:
            new_state_dict[f'embed_tokens.{name}'] += new_param

    del model
    torch.cuda.empty_cache()

    # 用差分更新 pre_model 的参数
    for name, value in new_state_dict.items():
        set_attr(
            pre_model,
            name.split('.'),
            nn.Parameter(value + pre_params[name], requires_grad=False)
        )

    # 仍然保留在 CPU，避免占 GPU
    return pre_model.to("cpu")



def load_single_merged_model_pre_llm(args, dataset):
    pre_model = load_pretrained_model(args).model
    check_gpu()
    del pre_model.norm, pre_model.layers
    check_gpu()

    new_state_dict = {}

    model = load_part_model(args, 'model.embed_tokens', args.task_model_mapping_dict[dataset])
    for name, param in model.named_parameters():
        new_param = (dict(model.named_parameters())[name] - dict(pre_model.named_parameters())[f'embed_tokens.{name}'])
        if new_state_dict.get(f'embed_tokens.{name}') is None:
            new_state_dict[f'embed_tokens.{name}'] = new_param
        else:
            new_state_dict[f'embed_tokens.{name}'] += new_param
    del model
    torch.cuda.empty_cache()

    for name, value in new_state_dict.items():
        set_attr(pre_model, name.split('.'), nn.Parameter(value + dict(pre_model.named_parameters())[name], requires_grad=False))

    return pre_model


def transform_data_loader_prelayer_pertask_llm(data_loader, merged_model, models, device, num_workers=0, shuffle=True, batch_size=1):
    transformed_data = []

    with torch.no_grad():
        for data in data_loader:
            x = data['data'][0].to(device)
            print(x['input_ids'].shape)
            source_loader = data['source_loader']

            inputs = []

            # model_output[0] is the input of the first layer, with shape [batch_size, seq_length, embedding_dim]
            model_output = merged_model(**x)
            inputs.append(model_output)
            model = models[source_loader.item()]
            model_output = model(**x)
            inputs.append(model_output)

            # shape of inputs: [num_tasks+1, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks+1, seq_length, embedding_dim]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()

            # batchsize = 1
            transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=num_workers)

    return new_dataloader



def transform_data_loader_layer_pertask_llm(data_loader, merged_model, models, device, 
                                            pre_causal_mask, pre_position_ids,args):
    transformed_data = []
    causal_mask = [pre_causal_mask[i].clone().to(args.device) for i in range(len(pre_causal_mask))]
                # causal_mask 是列表，每个元素都是 [batch, seq_len]
                # 这边的40就是默认的num_heads
    causal_mask_tensor_list = [mask[:, None, None, :].bool().expand(mask.shape[0], 40, mask.shape[1], mask.shape[1]) for mask in causal_mask]
    position_ids = [pre_position_ids[i].clone().to(args.device) for i in range(len(pre_position_ids))]
    with torch.no_grad():
        for data in data_loader:
            # shape of x: [batch_size, num_tasks+1, seq_length, embedding_dim] -> [num_tasks+1, batch_size, seq_length, embedding_dim]
            x = data['data'].to(device)
            x = x.permute(1, 0, 2, 3)

            source_loader = data['source_loader']

            inputs = []

            # output = merged_model(x[0], pre_causal_mask[0], pre_position_ids[0])[0]
            output = merged_model(x[0], causal_mask_tensor_list[0], position_ids[0])[0]
            inputs.append(output)
            idx = source_loader.item()
            model = models[idx]
            # output = model(x[1], pre_causal_mask[idx], pre_position_ids[idx])[0]
            output = model(x[1], causal_mask_tensor_list[idx], position_ids[idx])[0]
            inputs.append(output)

            # shape of inputs: [num_tasks, batch_size, seq_length, embedding_dim] -> [batch_size, num_tasks, seq_length, embedding_dim]
            inputs = torch.stack(inputs).permute(1, 0, 2, 3).cpu()

            # batchsize = 1
            transformed_data.append((inputs, source_loader))

    new_dataset = TransformedDataDataset(transformed_data)

    new_dataloader = DataLoader(new_dataset,
                                batch_size=1,
                                shuffle=True,
                                collate_fn=transformed_data_collate_fn,
                                num_workers=0)

    return new_dataloader


def load_fine_tuned_model_from_split(args, model_name):
    weight_map = get_weight_map_llm(model_name, args)
    model_path = os.path.join(os.path.join(args.cache_dir, model_name), f'{args.llm_version}/split')
    model = load_pretrained_model(args)
    # for name, param in model.named_parameters():
    #     print(name)
    for key, value in weight_map.items():
        weight_path = os.path.join(model_path, value)
        new_state_dict = torch.load(weight_path)
        model_key = get_attr(model, key.split('.'))
        model_key.load_state_dict(new_state_dict.state_dict())
    return model


def check_model_same(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        param1 = param1.cuda()
        param2 = param2.cuda()
        if name1 != name2:
            return False
        if not torch.equal(param1, param2):
            print(name1)
            print(torch.norm(param1 - param2))
            print(torch.norm(param1))
            print(torch.norm(param2))
            return False
        param1 = param1.cpu()
        param2 = param2.cpu()
    return True

def load_avg_merged_model_llm_whole(args, merge_coef=0.5):
    pre_model = load_pretrained_model(args)

    param_dict = dict(pre_model.named_parameters())

    for data_name in args.dataset_names:
        model = load_fine_tuned_model(args, data_name)
        for name, param in pre_model.named_parameters():
            new_param = (dict(model.named_parameters())[name]-dict(pre_model.named_parameters())[name]) * merge_coef
            param_dict[name] = param_dict[name] + new_param
        del model
        torch.cuda.empty_cache()
    
    for name, param in pre_model.named_parameters():
        set_attr(pre_model, name.split('.'), nn.Parameter(param_dict[name], requires_grad=False))

    return pre_model