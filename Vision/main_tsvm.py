# ---------------------------------------------------------------
# This script is adapted from “task_singular_vectors”
# Original repository: https://github.com/AntoAndGar/task_singular_vectors
# The code below (esp. compute_and_sum_svd_mem_reduction & build_task_vectors)
# follows the logic of the original open-source implementation,
# with minor modifications for our use case.
# ---------------------------------------------------------------
import torch
import torch.nn.functional as F
from src.eval import eval_single_dataset
from fda_args import parse_arguments
from utils import *
import numpy as np

import torch.nn.functional as F
from src.eval import eval_single_dataset
from utils import *

# Config
args = parse_arguments()
def compute_and_sum_svd_mem_reduction(task_vectors):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / 8
    device = "cuda"
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].keys():
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, dataset_list)):
                vec = task_vector[key].to(device)
                if (len(task_vector[key].shape) == 2 and "text_projection" not in key):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot((u_u,v_u,torch.diag(sum_s),u_v,v_v,))
    return new_vector


def build_task_vectors(model_name, pretrained_checkpoint, finetuned_checkpoints):
    """
    Build a list of NonLinearTaskVector objects from paired pretrained and finetuned checkpoints.

    Args:
        model_name (str): Model identifier (must be in MODELS list).
        pretrained_checkpoints (List[str]): List of file paths to pretrained checkpoints.
        finetuned_checkpoints (List[str]): List of file paths to fine-tuned checkpoints.

    Returns:
        List[NonLinearTaskVector]: Task vector objects.
    """

    task_vectors = []
    pretrained_state=torch.load(pretrained_checkpoint, map_location='cpu').state_dict()
    for ft in finetuned_checkpoints:
        task_vector={}
        finetuned_state=torch.load(finetuned_checkpoints[ft], map_location='cpu').state_dict()
        for key in finetuned_state.keys():
            task_vector[key] = finetuned_state[key]- pretrained_state[key]
        task_vectors.append(task_vector)

    return task_vectors

# Config
args.model='ViT-B-32'
args.model_location = f'{args.model_location}/{args.model}'
pretrained_checkpoint = f'{args.model_location}/zeroshot.pt'
dataset_list =['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB','MNIST','DTD']

finetuned_checkpoints = {
    dataset: f'{args.model_location}/{dataset}/finetuned.pt'
    for dataset in dataset_list
}

task_vectors = build_task_vectors(args.model, pretrained_checkpoint, finetuned_checkpoints)
new_merged_tv = compute_and_sum_svd_mem_reduction(task_vectors)

pretrained_state_dict=torch.load(pretrained_checkpoint, map_location='cpu').state_dict()

for key, delta in new_merged_tv.items():
    if key in pretrained_state_dict:
        pretrained_state_dict[key] += delta.to(pretrained_state_dict[key].device)
    else:
        print(f"Warning: {key} not found in pretrained state_dict.")

model=torch.load(pretrained_checkpoint)
model.load_state_dict(pretrained_state_dict)

metrics = {}
for dataset in dataset_list:
    metrics[dataset] = eval_single_dataset(model, dataset + 'ValfromTrain', args, dataset, None, None)['top1']
metrics['avg'] = sum(metrics.values()) / len(metrics)

print(metrics)


