from utils.utils import set_random_seed
from utils.customized_trainers import CustomizedTrainer
from utils.llm_data_loader import LLMDataLoader
from transformers import AutoTokenizer, TrainingArguments
import torch
import logging
import argparse
import sys
import os
import torch.nn.functional as F
import time
import sys
import numpy as np
import pprint 

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
finetuned_models = ["WizardLM-13B-V1.2", "WizardMath-13B-V1.0", "llama-2-13b-code-alpaca"]

parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--do_instruct", action="store_true", help="whether to merge instruct model")
parser.add_argument("--do_math", action="store_true", help="whether to merge math model")
parser.add_argument("--do_code", action="store_true", help="whether to merge code model")
parser.add_argument("--save_dual_anchors", action="store_true", help="whether to save anchors")
parser.add_argument("--language_model_name", type=str,
                    default="Llama-2-13b-hf", help="name of the language model")
parser.add_argument("--merging_method_name", type=str, default="sequential_efficient")
parser.add_argument("--merged_model_path", type=str, default="/data/shikexuan/save_merged_models_dare/math_code/math/mask_merging/task_arithmetic_scaling_coefficient_0.5/mask_0.9_0.9_rescale_True")
parser.add_argument("--name", type=str,default="init_by_params", help="name of the language model")
parser.add_argument("--val_shot", type=int, default=32,
                    help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--opt_steps", type=int, default=200, help="opt steps")
parser.add_argument("--opt_lr", type=float, default=1e-2, help="opt lr")
parser.add_argument("--anchor_num",type=int,default=1024)
parser.add_argument("--adapt_batch_size",type=int,default=1024)
parser.add_argument("--adapt_epochs",type=int,default=10)
parser.add_argument("--adapt_loss",type=str,default="cos")
parser.add_argument("--adapt_lr", type=float, default=1e-2, help="adapt lr")
parser.add_argument("--scale", type=float, default=0.01, help="scale")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
parser.add_argument("--random",action="store_true",help="Enable adaptment (default: False)")
parser.add_argument("--tag", type=str,default='test', help="tag for distill merging")
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

def train(args, lr, epochs, load_model_paths):
    num_layers = 40
    for layer_idx in range(num_layers): 
        print('\r Aligning',layer_idx)
        X_dict={}
        for dataset_name in args.dataset_names:
            finetuned_layer = load_part_model(args, f'model.layers.{layer_idx}', args.task_model_mapping_dict[dataset_name])
            finetuned_mlp=finetuned_layer.mlp
            anchors=np.load(f'/home/shikexuan/dual_anchors/{dataset_name}/{args.name}-scale-{args.scale}-opt-{args.opt_steps}-lr-{args.opt_lr}-anchor-{args.anchor_num}/dual_anchors_layer_{layer_idx+1}.npy')
            x=torch.from_numpy(anchors).to(torch.float32).cuda()
            gt=finetuned_mlp(x).detach().cpu()
            X_dict[dataset_name]={"input":x.detach().cpu(),'gt':gt}
        dataset = LabeledMultiDatasetFromDict(X_dict)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.adapt_batch_size, shuffle=True)

        mlps=[]
        mlp_pretrained = load_part_model(args, f'model.layers.{layer_idx}', args.language_model_name).mlp
        mlp_merged=load_part_model_for_merged(args,args.merged_model_path,f'model.layers.{layer_idx}').mlp
        mlps.append(mlp_merged)
        merged_mlps=MergedModel_FDAs(mlp_pretrained,mlps,'elementwise',1.0) 
        merged_mlps.train()
        optimizer=torch.optim.Adam(merged_mlps.parameters(),lr=args.adapt_lr)
        start_time=time.time()
        for epoch in range(args.adapt_epochs):
            epoch_loss = 0
            batch_count = 0
            for x,y in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                optimizer.zero_grad()
                out_merged=merged_mlps.get_merged_model()(x)
                if args.adapt_loss == 'cos':
                    loss = -F.cosine_similarity(out_merged, y, dim=1).sum()
                elif args.adapt_loss == 'mse':
                    loss = F.mse_loss(out_merged, y, reduction='sum')
                elif args.adapt_loss == 'l1':
                    loss = F.l1_loss(out_merged, y, reduction='sum')
                else:
                    raise ValueError(f"Unsupported adapt_loss type: {args.adapt_loss}")
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            avg_loss = epoch_loss / batch_count
            # 计算已用时间和预计剩余时间
            elapsed = time.time() - start_time
            epochs_done = epoch + 1
            avg_time_per_step = elapsed / epochs_done
            eta = avg_time_per_step * (args.adapt_epochs - epochs_done)
            print(f"\r[Epoch {epoch}/{args.adapt_epochs}] "f"Loss = {avg_loss:.4f} | ETA: {int(eta)}s", end="")
        merged_layer=load_part_model_for_merged(args,args.merged_model_path,f'model.layers.{layer_idx}')
        merged_layer.mlp = merged_mlps.get_merged_model()
        save_dir = f'/data/shikexuan/save_layers/{args.dataset_name_combined}/{args.name}-scale-{args.scale}/{args.language_model_name}-opt-{args.opt_steps}-opt-lr-{args.opt_lr}-anchor-{args.anchor_num}-adapt-{args.adapt_epochs}-lr-{args.adapt_lr}-bs-{args.adapt_batch_size}/'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(merged_layer, os.path.join(save_dir, f'layer_{layer_idx}.pt'))
        del merged_layer
        torch.cuda.empty_cache()
    
    merged_model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.merged_model_path, device_map='cpu')
    remove_grad(merged_model)
    for layer_idx in range(num_layers):
        merged_layer=torch.load(f'/data/shikexuan/save_layers/{args.dataset_name_combined}/{args.name}-scale-{args.scale}/{args.language_model_name}-opt-{args.opt_steps}-opt-lr-{args.opt_lr}-anchor-{args.anchor_num}-adapt-{args.adapt_epochs}-lr-{args.adapt_lr}-bs-{args.adapt_batch_size}/layer_{layer_idx}.pt',map_location='cpu')
        for name, _ in merged_model.model.layers[layer_idx].named_parameters():
            set_attr(merged_model.model.layers[layer_idx], name.split('.'), nn.Parameter(get_attr(merged_layer, name.split('.'))))
    
    return merged_model
    


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
            f"/data/shikexuan/MergeLM_models/{task_model_mapping_dict[dataset_name]}")

    args.save_merged_model_path = f"/data/shikexuan/save_merge_models/{args.dataset_name_combined}/{args.name}-scale-{args.scale}/{args.language_model_name}-opt-{args.opt_steps}-opt-lr-{args.opt_lr}-anchor-{args.anchor_num}-adapt-{args.adapt_epochs}-lr-{args.adapt_lr}-bs-{args.adapt_batch_size}/"
    args.load_model_paths_dict = {
        args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}
    
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")
    print("===== Current Configuration =====")
    pprint.pprint(vars(args))
    print("================================")
    
    merged_model=train(args, args.lr, args.epochs, load_model_paths)


    merged_model.save_pretrained(args.save_merged_model_path+"math/")
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.merged_model_path)
    tokenizer.save_pretrained(args.save_merged_model_path+"math/")


    parts = args.merged_model_path.split("/")
    idx = parts.index("math")
    parts[idx] = "code"
    new_path = "/".join(parts)

    merged_model.save_pretrained(args.save_merged_model_path+"code/")
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=new_path)
    tokenizer.save_pretrained(args.save_merged_model_path+"code/")
