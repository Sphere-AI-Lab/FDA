from utils.utils import set_random_seed
from utils.customized_trainers import CustomizedTrainer
from utils.llm_data_loader import LLMDataLoader
from transformers import AutoTokenizer, TrainingArguments
import torch
import logging
import argparse
import gc
import sys
import os
import tqdm
import torch.nn.functional as F
import shutil
import time
import sys
import numpy as np

from model_merging_methods.distill_merging_utils import *

cache_dir = "/home/shikexuan/MergeLM_models"

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

finetuned_models = [
    "WizardLM-13B-V1.2",
    "WizardMath-13B-V1.0",
    "llama-2-13b-code-alpaca"
]

parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--do_math", action="store_true", help="whether to merge math model")
parser.add_argument("--do_code", action="store_true", help="whether to merge code model")
parser.add_argument("--save_dual_anchors", action="store_true", help="whether to save anchors")
parser.add_argument("--language_model_name", type=str,default="Llama-2-13b-hf", help="name of the language model")
parser.add_argument("--name", type=str,default="init_by_params", help="name of the language model")
parser.add_argument("--val_shot", type=int, default=32,help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--gen_steps", type=int, default=1, help="gen steps")
parser.add_argument("--gen_lr", type=float, default=1e-2, help="gen lr")
parser.add_argument("--anchor_num",type=int,default=1024)
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
parser.add_argument("--random",action="store_true",help="Enable alignment (default: False)")
parser.add_argument("--scale",type=float,default=1)
parser.add_argument("--tag", type=str,
                    default='test', help="tag for distill merging")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--llm_version", type=str, default="v1.0", help="version of the language model")
try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available(
    ) and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"train_{args.val_shot}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def train(args):
    num_layers = 40
    for layer_idx in range(num_layers):
        print('\nStage 1: Obtain anchors from individual models for Layer ',layer_idx, 'Name', args.name, 'scale',args.scale, 'anchor num',args.anchor_num)
        X_dict = {}
        for dataset in args.dataset_names:
            pretrained_layer = load_part_model(args, f'model.layers.{layer_idx}', args.language_model_name)
            finetuned_layer = load_part_model(args, f'model.layers.{layer_idx}', args.task_model_mapping_dict[dataset])
            pretrained_mlp=pretrained_layer.mlp
            for param in pretrained_mlp.parameters():
                param.requires_grad = True
            finetuned_mlp=finetuned_layer.mlp 

            pretrained_params = [p for p in pretrained_mlp.parameters()]
            finetuned_params = [p for p in finetuned_mlp.parameters()]
            X=initialize_X(args,pretrained_params, finetuned_params)
            optimizer = torch.optim.AdamW([X], lr=args.gen_lr)
            best_loss = float('inf')
            best_X = None
            angle_loss=norm_diff=0.0
            start_time = time.time()
            for step in range(args.gen_steps):
                def closure():
                    nonlocal angle_loss,norm_diff,step
                    optimizer.zero_grad()
                    feat_pretrained = pretrained_mlp(X)
                    feat_finetuned = finetuned_mlp(X)
                    feature_loss = F.cosine_similarity(feat_pretrained,feat_finetuned,dim=1).mean()
                    grads = torch.autograd.grad(feature_loss,pretrained_params,create_graph=True,retain_graph=True)
                    grad_diff_loss = angle_loss = norm_diff =0.0  
                    for i, (g, theta_p, theta_f) in enumerate(zip(grads, pretrained_params, finetuned_params)):
                        vec = (theta_f - theta_p)
                        cos = torch.nn.functional.cosine_similarity(g.view(-1), (vec).view(-1), dim=0)
                        angle_loss+=1-cos
                    grad_diff_loss +=  angle_loss 
                    grad_diff_loss.backward()
                    nonlocal best_loss, best_X
                    if grad_diff_loss.item() < best_loss:
                        best_loss = grad_diff_loss.item()
                        best_X = X.detach().clone()
                    return grad_diff_loss 
                optimizer.step(closure)
                elapsed = time.time() - start_time
                steps_done = step + 1
                avg_time_per_step = elapsed / steps_done
                eta = avg_time_per_step * (args.gen_steps - steps_done)
                print(f"\r[Step {steps_done}/{args.gen_steps}]|{dataset} "f"angle_Loss = {angle_loss.item()/len(pretrained_params):.4f}|"f"ETA: {int(eta)}s", end="")
            X_dict[dataset] = {"input": best_X.detach().cpu(),"gt":gt}
            if args.save_dual_anchors:
                save_path1 = f'/home/shikexuan/dual_anchors/{dataset}/{args.name}-scale-{args.scale}-gen-{args.gen_steps}-lr-{args.gen_lr}-anchor-{args.anchor_num}'
                os.makedirs(save_path1, exist_ok=True)
                save_file = os.path.join(save_path1,f'dual_anchors_layer_{layer_idx+1}.npy')
                np.save(save_file, best_X.detach().cpu().numpy())            


if __name__ == "__main__":

    args.dataset_names = []
    if args.do_math:
        args.dataset_names.append("math")
    if args.do_code:
        args.dataset_names.append("code")
    args.dataset_name_combined = "_".join(args.dataset_names)
    args.cache_dir = cache_dir
    args.task_model_mapping_dict = task_model_mapping_dict
    args.finetuned_model_backbone_mapping_dict = finetuned_model_backbone_mapping_dict
    args.finetuned_models = finetuned_models

    set_random_seed(seed=0)

    load_model_paths = []
    for dataset_name in args.dataset_names:
        load_model_paths.append(
            f"/home/shikexuan/MergeLM_models/{task_model_mapping_dict[dataset_name]}")
    args.load_model_paths_dict = {
        args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}

    train(args)