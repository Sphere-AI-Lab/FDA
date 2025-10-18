import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
import torch
from fda_args import parse_arguments
from transformers import RobertaModel
import logging
from functools import partial
import os
import torch.nn.functional as F
import time
import numpy as np
from transformers import logging
logging.set_verbosity_error()
from model_merging_methods.distill_merging_utils import *
# 禁用高效注意力
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)


args = parse_arguments()

def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(args.seed) 


# save hyperparams
def save_hyperparams(args, filepath):
    lines = [
        "="*80,
        f"[Model]  arch: {args.model} | init params: {args.init_params} | device={args.device} |seed={args.seed}",
        f"[FDA settings]   anchor_num: {args.anchor_num} | token_num: {args.token_num} | init: {args.name} | scale={args.scale}",
        f"[Construction]   anchor_loss={args.anchor_loss} | optimize_steps={args.opt_steps} | lr={args.opt_lr} | save_anchors={args.save_dual_anchors}",
        "="*80,
    ]
    content = "\n".join(lines)
    print(content)
    os.makedirs(filepath, exist_ok=True)
    with open(filepath+'/construct_settings.text', "w") as f:
        f.write(content)

def train(args):
    if args.model=="roberta-large":
        num_layers=24
        args.alpha=0.2
    else:
        num_layers=12
        args.alpha=0.3

    print('Start training')
    print(args.init_params)
    for layer_idx in range(num_layers):
        print('\nStage1: Obtain feats from individ model for Layer ',layer_idx)
        X_dict = {}
        attention_mask=None
        for dataset in args.dataset_names:
            pretrained_encoder=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_checkpoint).to(args.device)
            finetuned_encoder=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.load_model_paths_dict[dataset]).to(args.device)
            pretrained_layer=pretrained_encoder.roberta.encoder.layer[layer_idx]
            finetuned_layer=finetuned_encoder.roberta.encoder.layer[layer_idx]
            pretrained_params = [p for p in pretrained_layer.parameters()]
            finetuned_params = [p for p in finetuned_layer.parameters()]
        
            if 'pos' in args.init_params:
                position_ids = torch.arange(args.token_num, device=args.device).unsqueeze(0).expand(args.anchor_num, -1)  # [B, L]
                position_emb = finetuned_encoder.roberta.embeddings.position_embeddings(position_ids).detach()  # [B, L, D] 
            else:
                position_emb=None
            X,attention_mask=initialize_X(args, params=finetuned_params)
            
            del pretrained_encoder, finetuned_encoder
            torch.cuda.empty_cache()

            optimizer = torch.optim.AdamW([X], lr=args.opt_lr)
            start_time = time.time()
            num_steps=args.opt_steps
            best_loss = float('inf')
            best_X = None
            gt=None
            angle_loss=0.0
            for step in range(num_steps):
                def closure():
                    nonlocal angle_loss
                    optimizer.zero_grad()
                    # x: [anchor_num,seq_length,embedding_dim]
                    feat_finetuned = finetuned_layer(X + position_emb if position_emb is not None else X,attention_mask,None,None,None,None,False)[0] # [batch_size,seq_length,embedding_dim]
                    feat_pretrained = pretrained_layer(X + position_emb if position_emb is not None else X,attention_mask,None,None,None,None,False)[0] # [batch_size,seq_length,embedding_dim]
                    if args.anchor_loss=='cos':
                        feature_loss=F.cosine_similarity(feat_pretrained.reshape(feat_pretrained.shape[0],-1),feat_finetuned.reshape(feat_finetuned.shape[0],-1),dim=1).mean()
                    if args.anchor_loss=='mse':
                        feature_loss=-F.mse_loss(feat_pretrained,feat_finetuned,reduction='mean')
                    if args.anchor_loss=='l1':
                        feature_loss=-F.l1_loss(feat_pretrained,feat_finetuned,reduction='mean')
                    grads = torch.autograd.grad(feature_loss,pretrained_params,create_graph=True,retain_graph=True)
                    grad_diff_loss = 0
                    angle_loss=0
                    for g, theta_p, theta_f in zip(grads, pretrained_params, finetuned_params):
                        vec = (theta_f - theta_p)
                        cos = torch.nn.functional.cosine_similarity(g.view(-1), (vec).view(-1), dim=0)
                        angle_loss+=1-cos
                    grad_diff_loss +=  angle_loss
                    grad_diff_loss.backward()
                    nonlocal best_loss, best_X, gt
                    if grad_diff_loss.item() < best_loss:
                        best_loss = grad_diff_loss.item()
                        best_X = X.detach().clone()
                        gt=feat_finetuned.detach().clone()
                    return grad_diff_loss
                optimizer.step(closure)    
                elapsed = time.time() - start_time
                steps_done = step + 1
                avg_time_per_step = elapsed / steps_done
                eta = avg_time_per_step * (num_steps - steps_done)
                print(f"\r[Step {steps_done}/{num_steps}]|{dataset} "f"angle_Loss = {angle_loss.item()/12:.4f}| ETA: {int(eta)}s", end="")
            print()
            X_dict[dataset] = {"input": (best_X + position_emb if position_emb is not None else best_X).detach().cpu(),"gt":gt,"attention_mask":torch.zeros(args.anchor_num,1,args.token_num,args.token_num).cpu()}
            if args.save_dual_anchors:
                save_path=f"{args.save_anchors_path}/{dataset}"
                os.makedirs(save_path, exist_ok=True)
                numpy_dict = {key: value.detach().cpu().numpy() for key, value in X_dict[dataset].items()}
                save_path=os.path.join(save_path,f'dual_anchors_layer_{layer_idx+1}.npy')
                np.save(save_path, numpy_dict)
    return None



if __name__ == "__main__":
    args.dataset_names = ["cola", "sst2", "mrpc","stsb", "qqp", "mnli", "qnli", "rte"]
    
    dataset_model_learning_rate_mapping_dict = {"cola_roberta-base": 1e-5,
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

    pretrained_checkpoint = f'{args.model_location}/{args.model}'

    load_model_paths = []
    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.model}"]
        load_model_paths.append(f"{args.model_location}/{dataset_name}/{args.model}_lr{learning_rate}")
    args.load_model_paths_dict = {args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}
    
    if 'weights' in args.name:
        args.save_anchors_path = f'{args.save_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-1.0-opt-{args.opt_steps}-{args.opt_lr}'
    elif 'random' in args.name:
        args.save_anchors_path= f'{args.save_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-{args.scale}-opt-{args.opt_steps}-{args.opt_lr}'
    else:
        raise ValueError(f'Invalid init method: {args.name}')

    save_hyperparams(args, args.save_anchors_path)
    train(args)

