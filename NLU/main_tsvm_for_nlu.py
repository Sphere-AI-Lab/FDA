import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from fda_args import parse_arguments
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.customized_trainers import CustomizedTrainer
from utils.metrics import compute_metrics
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import numpy as np
from functools import partial
import torch.nn.functional as F
from model_merging_methods.distill_merging_utils import *
from datasets import logging as datasets_logging
import torch.nn.functional as F
from torch.utils.data import DataLoader

def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

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
    # sv_reduction = 1 / len(config.DATASETS)
    sv_reduction = 1 / len(args.dataset_names)
    # device = config.device
    device = "cuda"
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].keys():
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, args.dataset_names)):
                vec = task_vector[key].to(device)
                if (len(task_vector[key].shape) == 2 and "text_projection" not in key and 'classifier' not in key):
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
                    elif 'classifier' not in key:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector[key].shape) == 2 and "text_projection" not in key and 'classifier' not in key:
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

def evaluate(args, merged_model, classifier_heads, eval_datasets, tokenizer, logger):
    acc = {}
    for idx, (dataset_name, classifier, eval_dataset) in enumerate(zip(args.dataset_names, classifier_heads, eval_datasets)):
        # since the classifier is not merged, we additionally set the classifier of merged_model for each model_to_merge
        merged_model.classifier = classifier

        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=TrainingArguments(
                args.save_merged_model_path,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
            ),
            eval_dataset=eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[
                                    dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )
        merged_model.eval()

        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(
            v, float) else v for k, v in test_metrics.items()}
        logger.info(
            f"test performance on dataset {dataset_name}: {test_metrics}")
        acc[dataset_name] = test_metrics[f'eval_{glue_data_metrics_map[dataset_name]}']

    acc['avg'] = sum(acc.values()) / len(acc)
    print(acc['avg'])

if __name__ == '__main__':
    args = parse_arguments()

    seed_torch(args.seed) 
    
    args.dataset_names = ["cola", "sst2", "mrpc",
                          "stsb", "qqp", "mnli", "qnli", "rte"]
    dataset_model_learning_rate_mapping_dict = {
        "cola_roberta-base": 1e-5,
        "sst2_roberta-base": 1e-5,
        "mrpc_roberta-base": 1e-5,
        "stsb_roberta-base": 1e-5,
        "qqp_roberta-base": 1e-5,
        "mnli_roberta-base": 1e-5,
        "qnli_roberta-base": 1e-5,
        "rte_roberta-base": 1e-5,
        "cola_roberta-large": 1e-5,
        "sst2_roberta-large": 1e-5,
        "mrpc_roberta-large": 1e-5,
        "stsb_roberta-large": 1e-5,
        "qqp_roberta-large": 1e-5,
        "mnli_roberta-large": 1e-5,
        "qnli_roberta-large": 1e-5,
        "rte_roberta-large": 1e-5
    }
    load_model_paths = []
    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.model}"]
        load_model_paths.append(f"{args.model_location}/{dataset_name}/{args.model}_lr{learning_rate}")

    args.load_model_paths_dict = {args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}

    pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=f'{args.model_location}/{args.model}').to(args.device)
    pretrained_state_dict = pretrained_model.state_dict()
    task_vectors=[]
    for dataset_name in args.dataset_names:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.load_model_paths_dict[dataset_name]).to(args.device)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            if 'classifier' not in key:
                state_dict[key] = state_dict[key] - pretrained_state_dict[key]
        task_vectors.append(state_dict)

    new_merged_tv = compute_and_sum_svd_mem_reduction(task_vectors)
    
    new_statedict={}
    for key in pretrained_state_dict.keys():
        if key not in new_merged_tv.keys():
            continue
        new_statedict[key]=pretrained_state_dict[key]+args.alpha*new_merged_tv[key]
    del model, state_dict, task_vectors
    torch.cuda.empty_cache()

    merged_model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=f'{args.model_location}/{args.model}').to(args.device)
    merged_model.load_state_dict(new_statedict,strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=f'{args.model_location}/{args.model}')
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)
    metrics = {}  # Dictionary to store the results for each dataset
    for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):
        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(
            dataset_name=dataset_name,
            val_shot_from_train=None,
            max_seq_length=128,
            seed=args.seed
        )  # type: ignore
        model_to_merge = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=load_model_path,num_labels=num_labels).to(args.device)
        # Set classifier to the model's classifier
        merged_model.classifier = model_to_merge.classifier

        merged_model_evaluator = CustomizedTrainer(model=merged_model,  # final merged model
            args=TrainingArguments(
                args.save_merged_model_path,
                per_device_train_batch_size=args.adapt_batch_size,
                per_device_eval_batch_size=args.adapt_batch_size,
            ),
            eval_dataset=test_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer)  # tokenizer
        merged_model.eval()
        # Evaluate the model
        test_metrics = merged_model_evaluator.evaluate()

        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        # Print result for each dataset
        acc_value = test_metrics[f'eval_{glue_data_metrics_map[dataset_name]}']
        print(f"{dataset_name} acc: {acc_value}")
        # Store the result in the metrics dictionary
        metrics[dataset_name] = acc_value
    avg_acc = sum(metrics.values()) / len(metrics)
    print(f"\nAverage accuracy across {len(metrics)} datasets: {avg_acc:.4f}")