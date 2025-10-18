from utils.utils import set_random_seed
from transformers import AutoTokenizer
import torch
import argparse
import sys
import os

from model_merging_methods.distill_merging_utils import *

cache_dir = "/home/shikexuan/MergeLM_models"

os.environ["WANDB_DISABLED"] = "true"


task_model_mapping_dict = {
    "instruct": "WizardLM-13B-V1.2",
    "math": "WizardMath-13B-V1.0",
    "code": "llama-2-13b-code-alpaca"
}

finetuned_model_backbone_mapping_dict = {
    "WizardLM-13B-V1.2": "Llama-2-13b-hf",
    "WizardMath-13B-V1.0": "Llama-2-13b-hf",
    "llama-2-13b-code-alpaca": "Llama-2-13b-hf"
}
finetuned_models = ["WizardLM-13B-V1.2", "WizardMath-13B-V1.0", "llama-2-13b-code-alpaca"]


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--language_model_name", type=str,
                    default="Llama-2-13b-hf_32001", help="name of the language model")
parser.add_argument("--merging_method_name", type=str,
                    default="average", help="name of the merging method")
parser.add_argument("--val_shot", type=int, default=16,
                    help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")

parser.add_argument("--tag", type=str,
                    default='test', help="tag for distill merging")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--layer_save", type=str, default="./save_layers", help="path to save layers in merging")
parser.add_argument("--llm_version", type=str, default="v1.0", help="version of the language model")
try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available(
    ) and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()

# args.dataset_names = ["instruct", "math", "code"]
args.dataset_names = ["math", "code"]
args.dataset_name_combined = "_".join(args.dataset_names)
args.cache_dir = cache_dir
args.task_model_mapping_dict = task_model_mapping_dict
args.finetuned_model_backbone_mapping_dict = finetuned_model_backbone_mapping_dict
args.finetuned_models = finetuned_models

args.device = 'cpu'

set_random_seed(seed=0)

load_model_paths = []
for dataset_name in args.dataset_names:
    # best checkpoint setting
    load_model_paths.append(
        f"/home/shikexuan/MergeLM_models/{task_model_mapping_dict[dataset_name]}")

args.save_merged_model_path = f"./save_merge_models/{args.dataset_name_combined}_0.5/{args.merging_method_name}/{args.language_model_name}/{args.llm_version}"
args.load_model_paths_dict = {
    args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}

try:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(os.path.join(cache_dir, args.language_model_name), args.llm_version))
except:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)

print('Merging...')
merged_model = load_avg_merged_model_llm(args, merge_coef=0.5)
os.makedirs(
        args.save_merged_model_path, exist_ok=True)

merged_model.save_pretrained(args.save_merged_model_path)
tokenizer.save_pretrained(args.save_merged_model_path)