# ---------------------------------------------------------------
# This script is adapted from "WUDI-Merging”
# Original repository: https://github.com/nathanielyvo/WUDI-Merging
# The code below follows the logic of the original open-source implementation,
# with minor modifications for our use case.
# ---------------------------------------------------------------
import torch
import torch.nn.functional as F
from src.eval import eval_single_dataset
from fda_args import parse_arguments
from utils import *
from copy import deepcopy
import random
import numpy as np


args = parse_arguments()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_redundant_task_vector(key, vectors, iter_num = 300, ratio = 1):

    vectors = vectors.cuda() # vectors.shape = [8, 2304, 768]

    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim = 0)))

    optimizer = torch.optim.Adam([merging_vector], lr=1e-5, weight_decay= 0)

    l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim= -1))

    for i in range(iter_num):
        disturbing_vectors = merging_vector.unsqueeze(0)- vectors
        inner_product = torch.matmul(disturbing_vectors , vectors.transpose(1,2)) 

        loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1) )
        optimizer.zero_grad()          
        loss.backward()
        optimizer.step()

    return merging_vector.data.detach().cpu()


def decompose_task_vectors(task_vectors, iter_num = 300, ratio = 1):

    merged_task_vector = {}
    keys = [key for key in task_vectors[0].keys() if ('attn.in_proj_weight' in key or 'attn.out_proj.weight' in key or 'mlp.c_fc.weight' in key or 'mlp.c_proj.weight' in key)]
    for key in keys:
        print(key)
        merged_task_vector[key] = torch.zeros_like(task_vectors[0][key])
        values = deepcopy(torch.stack([task_vector[key] for task_vector in task_vectors]))
        merigng_vector = get_redundant_task_vector(key, values, iter_num, ratio = ratio)
        merged_task_vector[key] += merigng_vector

    return merged_task_vector

setup_seed(0)
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
args.model='ViT-B-32'
args.model_location = f'{args.model_location}/{args.model}'
pretrained_checkpoint= f'{args.model_location}/zeroshot.pt'
pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

task_vectors=[]
for dataset_name in exam_datasets:
    state_dict=torch.load(f'{args.model_location}/{dataset_name}/finetuned.pt').state_dict()
    for key in state_dict.keys():
        state_dict[key]=state_dict[key]-pretrained_state_dict[key]
    task_vectors.append(state_dict)
task_vector = decompose_task_vectors(task_vectors, iter_num = 300, ratio = 1)
new_statedict={}
for key in pretrained_state_dict.keys():
    if key not in task_vector.keys():
        continue
    new_statedict[key]=pretrained_state_dict[key]+task_vector[key]

model=torch.load(pretrained_checkpoint)
model.load_state_dict(new_statedict,strict=False)
metrics = {}
for dataset in dataset_list:
    metrics[dataset] = eval_single_dataset(model, dataset + 'ValfromTrain', args, dataset, None, None)['top1']
metrics['avg'] = sum(metrics.values()) / len(metrics)
print(metrics)

    