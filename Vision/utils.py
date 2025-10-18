import torch
import copy
from itertools import combinations
from torch.utils.data import Dataset
import torch
import torch.nn as nn

dataset_list = ['MNIST', 'EuroSAT', 'GTSRB',
                'SVHN', 'DTD', 'RESISC45', 'Cars', 'SUN397']

def load_merged_layers(args, layer_idx):
    pretrained_checkpoint = f'{args.model_location}/zeroshot.pt'
    image_encoder_pretrained = torch.load(pretrained_checkpoint, map_location=args.device)
    layer_pretrained = image_encoder_pretrained.model.visual.transformer.resblocks[layer_idx]

    if args.init_params=='ta':
        layers = []
        for dataset in dataset_list:
            image_encoder = torch.load(
                f'{args.model_location}/{dataset}/finetuned.pt', map_location=args.device)
            layer = image_encoder.model.visual.transformer.resblocks[layer_idx]
            layers.append(layer)
            del image_encoder
            torch.cuda.empty_cache()
        merged_layers = MergedModel(layer_pretrained, layers,init_scale=0.3)
        return merged_layers
    
    elif args.init_params=='tsv':
        merged_encoder=torch.load(f'{args.model_location}/merged_tsv.pt',map_location=args.device)
        merged_layer=merged_encoder.model.visual.transformer.resblocks[layer_idx]
        layer_merged=[]
        layer_merged.append(merged_layer)
        del merged_encoder
        torch.cuda.empty_cache()
        merged_layers = MergedModel(layer_pretrained, layer_merged,init_scale=1.0)
        return merged_layers
    
    elif args.init_params=='wudi':
        merged_encoder=torch.load(f'{args.model_location}/merged_wudi.pt',map_location=args.device)
        merged_layer=merged_encoder.model.visual.transformer.resblocks[layer_idx]
        layer_merged=[]
        layer_merged.append(merged_layer)
        del merged_encoder
        torch.cuda.empty_cache()
        merged_layers = MergedModel(layer_pretrained, layer_merged,init_scale=1.0)
        return merged_layers

class LabeledMultiDatasetFromDict(Dataset):
    def __init__(self, X_dict):
        """
        X_dict: {
            'dataset_name': {
                'input': Tensor [seq_len, batch_size, hidden_dim],
                'label': Tensor [seq_len, batch_size, hidden_dim]
            }, ...
        }
        """
        self.inputs = []
        self.labels = []
        for name, pair in X_dict.items():
            X = pair["input"].permute(1, 0, 2)   # [batch_size, seq_len, hidden_dim]
            Y = pair["gt"].permute(1, 0, 2)   # [batch_size, seq_len, hidden_dim]
            self.inputs.extend(X)
            self.labels.extend(Y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def to_float(val):
    return val.item() if isinstance(val, torch.Tensor) else val

def initialize_X(args, params):
    device = args.device

    if args.model == 'ViT-L-14':
        feature_dim = 1024
        token_num = 257
    elif args.model == 'ViT-B-16':
        feature_dim = 768
        token_num = 197
    elif args.model == 'ViT-B-32':
        feature_dim = 768
        token_num = 50
    
    if args.name=='init_by_gauss_random':
        torch.manual_seed(0) 
        if args.model == 'ViT-L-14':
            X = torch.randn(257, args.anchor_num, 1024, device=device)
        elif args.model == 'ViT-B-16':
            X = torch.randn(197, args.anchor_num, 768, device=device) 
        elif args.model == 'ViT-B-32':
            X = torch.randn(50, args.anchor_num, 768, device=device)
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        X=X*args.scale
        X.requires_grad_()
    if args.name=='init_by_weights':
        if args.model == 'ViT-L-14':
            feature_dim = 1024
            token_num = 257
        elif args.model == 'ViT-B-16':
            feature_dim = 768
            token_num = 197
        elif args.model == 'ViT-B-32':
            feature_dim = 768
            token_num = 50
        all_features = []
        param_features=[]
        for p in params:
            if p.ndim == 2 and p.shape[1] == feature_dim:
                param_features.append(p.detach().reshape(-1, feature_dim))
        all_features=torch.cat(param_features,dim=0)
        if all_features.shape[0] >= token_num * args.anchor_num:
            idx = torch.randperm(all_features.shape[0])[:token_num * args.anchor_num]
            X = all_features[idx].reshape(token_num, args.anchor_num, feature_dim).to(device)
            X.requires_grad_() 
        else:
            needed = token_num * args.anchor_num - all_features.shape[0]
            # 先把已有的样本都放进去
            existing = all_features
            # 补齐部分
            extra = []
            for _ in range(needed):
                i, j = torch.randint(0, all_features.shape[0], (2,))
                mean_vec = (all_features[i] + all_features[j]) / 2
                extra.append(mean_vec.unsqueeze(0))
            extra = torch.cat(extra, dim=0)
            # 拼接后 reshape
            all_padded = torch.cat([existing, extra], dim=0)
            X = all_padded[:token_num * args.anchor_num].reshape(token_num, args.anchor_num, feature_dim).to(device)
            X.requires_grad_()
    return X


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



class MergedModel(nn.Module):
    def __init__(self, pretrained_model, models, init_scale=0.3):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.models = models
        self.init_scale = init_scale

        # freeze pretrained parameters
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        for m in self.models:
            for p in m.parameters():
                p.requires_grad = False

        self.alphas = nn.ParameterList()
        for model in self.models:
            alpha_list = nn.ParameterList()
            for p in model.parameters():
                alpha_list.append(
                    nn.Parameter(torch.ones_like(p) * init_scale)
                )
            self.alphas.append(alpha_list)

        self.merged_model = copy.deepcopy(self.pretrained_model)
        _, self.names = make_functional(self.merged_model)

    def get_merged_model(self):
        merged_param = []
        for idx, (name, p0) in enumerate(self.pretrained_model.named_parameters()):
            p_merge = torch.zeros_like(p0)
            for k in range(len(self.models)):
                p_k = dict(self.models[k].named_parameters())[name]
                alpha = self.alphas[k][idx]
                p_merge += alpha * (p_k - p0)
            p_merge += p0
            merged_param.append(p_merge)

        load_weights(self.merged_model, self.names, merged_param)
        return self.merged_model

    def forward(self, x):
        return self.get_merged_model()(x)