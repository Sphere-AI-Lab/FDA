import torch # type: ignore
import os
import torch.nn.functional as F # type: ignore
from fda_args import parse_arguments
from utils import *
import random
import numpy as np # type: ignore
import time

# To compute the second-order gradients
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# For reproducibility
def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# Set the hyperparameters and paths
args = parse_arguments()
args.model_location = f'{args.model_location}/{args.model}'
pretrained_checkpoint = f'{args.model_location}/zeroshot.pt'

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

seed_torch(args.seed) 

# The construction process of FDAs
def train(args):
    if args.model in ['ViT-B-32', 'ViT-B-16']:
        num_layers = 12
    elif args.model in ['ViT-L-14']:
        num_layers = 24
    else:
        raise ValueError(f'Invalid model: {args.model}')
    dataset_loss={}
    for layer_idx in range(num_layers):
        print('\nConstruct anchors from individual models for Layer',layer_idx)
        for dataset in dataset_list:
            # Read Encoders
            pretrained_encoder=torch.load(f'{args.model_location}/zeroshot.pt', map_location=args.device)
            finetuned_encoder= torch.load(f'{args.model_location}/{dataset}/finetuned.pt', map_location=args.device)
            finetuned_layer=finetuned_encoder.model.visual.transformer.resblocks[layer_idx]
            pretrained_layer=pretrained_encoder.model.visual.transformer.resblocks[layer_idx]
            finetuned_params = [p for p in finetuned_layer.parameters()]
            pretrained_params = [p for p in pretrained_layer.parameters()]
            
            # The detailed derivation for initialization can be found in our paper.
            X=initialize_X(args=args,params=finetuned_params)
            del pretrained_encoder,finetuned_encoder
            torch.cuda.empty_cache()

            optimizer = torch.optim.AdamW([X], lr=args.opt_lr)
            start_time = time.time()
            best_loss = float('inf')
            best_X = None
            angle_loss=norm_diff=0.0

            # Optimize Anchors
            for step in range(args.opt_steps):
                def closure():
                    nonlocal angle_loss,norm_diff,step, best_loss, best_X
                    optimizer.zero_grad()
                    feat_pretrained = pretrained_layer(X).permute(1, 0, 2).reshape(X.shape[1],-1) 
                    feat_finetuned = finetuned_layer(X).permute(1, 0, 2).reshape(X.shape[1],-1) 
                    if args.anchor_loss=='cos':
                        feature_loss = F.cosine_similarity(feat_pretrained,feat_finetuned,dim=1).mean()
                    if args.anchor_loss=='mse':
                        feature_loss = -F.mse_loss(feat_pretrained,feat_finetuned,reduction='mean')
                    if args.anchor_loss == 'l1':
                        feature_loss = -F.l1_loss(feat_pretrained, feat_finetuned, reduction='mean')
                    grads = torch.autograd.grad(feature_loss,pretrained_params,create_graph=True,retain_graph=True) # type: ignore
                    grad_diff_loss = angle_loss = norm_diff  =0.0  
                    for i, (g, theta_p, theta_f) in enumerate(zip(grads, pretrained_params, finetuned_params)):
                        vec = theta_f - theta_p
                        cos = torch.nn.functional.cosine_similarity(g.view(-1), (vec).view(-1), dim=0)
                        angle_loss+=1-cos
                        norm_diff += (g.norm() - vec.norm()).pow(2)
                    grad_diff_loss +=  angle_loss 
                    grad_diff_loss.backward() # type: ignore
                    if grad_diff_loss.item() < best_loss: # type: ignore
                        best_loss = grad_diff_loss.item() # type: ignore
                        best_X = X.detach().clone()
                    return grad_diff_loss 
                optimizer.step(closure)    
                elapsed = time.time() - start_time
                steps_done = step + 1
                avg_time_per_step = elapsed / steps_done
                eta = avg_time_per_step * (args.opt_steps - steps_done)
                print(f"\r[Step {steps_done}/{args.opt_steps}]|{dataset}" f"angle_Loss = {angle_loss.item()/len(pretrained_params):.4f}|" f"norm_diff = {to_float(norm_diff)/len(pretrained_params):.4f}|" f"ETA: {int(eta)}s", end="")
            print()
            dataset_loss[str(layer_idx)+dataset]={"angle_loss":f"{best_loss//len(pretrained_params):.4f}"}
            if args.save_dual_anchors:
                save_path = f'{args.save_anchors_path}/{dataset}'
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path,f'dual_anchors_layer_{layer_idx+1}.npy')
                np.save(save_file, best_X.cpu().numpy())
    return None


if __name__ == '__main__':
    # Default save path
    if args.name=='init_by_weights':
        args.save_anchors_path = f'{args.save_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-1.0-opt-{args.opt_steps}-{args.opt_lr}'
    elif args.name=='init_by_gauss_random':
        args.save_anchors_path= f'{args.save_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-{args.scale}-opt-{args.opt_steps}-{args.opt_lr}'
    else:
        raise ValueError(f'Invalid init method: {args.name}')
    # Or you can define any save_path you want    
    save_hyperparams(args, args.save_anchors_path)
    train(args)
