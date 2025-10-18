import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from utils.load_config import cache_dir
from utils.customized_trainers import CustomizedTrainer
from utils.metrics import compute_metrics
from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import torch
from functools import partial
from fda_args import parse_arguments
import os
import torch.nn.functional as F
import numpy as np
from model_merging_methods.distill_merging_utils import *
import torch
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
import random
import numpy as np

def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_redundant_task_vector(key, vectors, iter_num = 300, ratio = 1):


    vectors = vectors.cuda() # vectors.shape = [8, 2304, 768]

    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim = 0)))

    optimizer = torch.optim.Adam([merging_vector], lr=2e-5)

    l2_norms = torch.square(torch.norm(vectors.reshape(8, -1), p=2, dim=1))

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
    import re
    exclude_patterns = [
        ".*classifier.*",
        ".*bias.*",
        ".*LayerNorm.*",
        ".*embeddings.*"
    ]

    keys = [
        key for key in task_vectors[0].keys()
        if not any(re.match(pat, key) for pat in exclude_patterns)
    ]
    for key in keys:
        merged_task_vector[key] = torch.zeros_like(task_vectors[0][key])
        values = deepcopy(torch.stack([task_vector[key] for task_vector in task_vectors]))
        merigng_vector = get_redundant_task_vector(key, values, iter_num, ratio = ratio)
        merged_task_vector[key] += merigng_vector.cuda()
    
    return merged_task_vector

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

# CUDA_VISIBLE_DEVICES=7 python main_wudi_for_nlu.py --model roberta-base --alpha 0.7
# CUDA_VISIBLE_DEVICES=7 python main_wudi_for_nlu.py --model roberta-base --alpha 0.5

if __name__ == '__main__':
    args = parse_arguments()

    seed_torch(args.seed) 
    
    args.dataset_names = ["cola", "sst2", "mrpc","stsb", "qqp", "mnli", "qnli", "rte"]
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
                state_dict[key] = state_dict[key] - pretrained_state_dict[key].clone()
        task_vectors.append(state_dict)
    task_vector = decompose_task_vectors(task_vectors, iter_num = 300, ratio = 1)

    new_statedict={}
    for key in pretrained_state_dict.keys():
        if key not in task_vector.keys():
            continue
        new_statedict[key]=pretrained_state_dict[key]+args.alpha*task_vector[key]
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
    print('alpha',args.alpha)


