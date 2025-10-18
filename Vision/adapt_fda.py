import torch
import os
import torch.nn.functional as F
from src.eval import eval_single_dataset
from fda_args import parse_arguments
from utils import *
import random
import numpy as np
import time

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def seed_torch(seed): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# Config
args = parse_arguments()
args.model_location = f'{args.model_location}/{args.model}'
pretrained_checkpoint = f'{args.model_location}/zeroshot.pt'

seed_torch(args.seed) 

def save_hyperparams(args, filepath):
    lines = [
        "="*80,
        f"[Model]  arch: {args.model} | init params: {args.init_params} | device={args.device} |seed={args.seed}",
        f"[FDA settings]   anchor_num: {args.anchor_num} | token_num: {args.token_num} | init: {args.name} | scale={args.scale}",
        f"[Construction]   anchor_loss={args.anchor_loss} | optimize_steps={args.opt_steps} | lr={args.opt_lr} | save_anchors={args.save_dual_anchors}",
        f"[Adapt]     epochs={args.adapt_epochs} | batch size={args.adapt_batch_size} | lr={args.adapt_lr} | loss={args.adapt_loss}",
        "="*80,
    ]
    content = "\n".join(lines)
    print(content)
    os.makedirs(filepath, exist_ok=True)
    with open(filepath+'/adaptation_settings.text', "w") as f:
        f.write(content)


def train(args):
    if args.model in ['ViT-B-32', 'ViT-B-16']:
        num_layers = 12
    elif args.model in ['ViT-L-14']:
        num_layers = 24
    else:
        raise ValueError(f'Invalid model: {args.model}')
    merged_model=torch.load(f'{args.model_location}/zeroshot.pt', map_location='cpu')
    print(args.name)
    for layer_idx in range(num_layers):
        print('\nStage 1: Read saved anchors for layer',layer_idx)
        X_dict = {}
        # Gather all anchors in X_dict
        for dataset in dataset_list:
            finetuned_encoder= torch.load(f'{args.model_location}/{dataset}/finetuned.pt', map_location=args.device)
            finetuned_layer=finetuned_encoder.model.visual.transformer.resblocks[layer_idx]
            read_file=os.path.join(args.read_path,f"{dataset}",f'dual_anchors_layer_{layer_idx+1}.npy')
            anchors = np.load(read_file)
            best_X= torch.from_numpy(anchors).to(torch.float32).cuda()
            gt = finetuned_layer(best_X).detach().cpu()
            X_dict[dataset] = {"input": best_X.detach().cpu(),"gt":gt}
        
        print('Stage 2: Adapt the layer',layer_idx)
        # Mix anchors of different tasks
        dataset = LabeledMultiDatasetFromDict(X_dict)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.adapt_batch_size, shuffle=True)
        if args.init_params=='pretrained':
            merged_layer=merged_model.model.visual.transformer.resblocks[layer_idx].cuda()
        elif args.init_params in ['ta','tsv','wudi']:
            merged_layer= load_merged_layers(args, layer_idx)
        else:
            raise ValueError(f'Invalid init_params: {args.init_params}')
        merged_layer.train()
        optimizer=torch.optim.Adam(merged_layer.parameters(), lr=args.adapt_lr)
        start_time=time.time()
        batch_count = 0
        epoch_loss=0

        for epoch in range(args.adapt_epochs):
            epoch_loss = 0
            batch_count = 0
            for x,y in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                optimizer.zero_grad()
                if args.init_params in ['ta','tsv','wudi']:
                    out_merged=merged_layer.get_merged_model()(x.permute(1,0,2)).permute(1,0,2).reshape(args.adapt_batch_size, -1)
                elif args.init_params=='pretrained':
                    out_merged=merged_layer(x.permute(1,0,2)).permute(1,0,2).reshape(args.adapt_batch_size, -1)
                else:
                    raise ValueError(f'Invalid init_params: {args.init_params}')
                
                if args.adapt_loss == 'cos':
                    loss = -F.cosine_similarity(out_merged, y.reshape(args.adapt_batch_size, -1), dim=1).sum()
                elif args.adapt_loss == 'mse':
                    loss = F.mse_loss(out_merged, y.reshape(args.adapt_batch_size, -1), reduction='sum')
                elif args.adapt_loss == 'l1':
                    loss = F.l1_loss(out_merged, y.reshape(args.adapt_batch_size, -1), reduction='sum')
                else:
                    raise ValueError(f"Unsupported align_loss type: {args.adapt_loss}")
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            avg_loss = epoch_loss / batch_count
            elapsed = time.time() - start_time
            epochs_done = epoch + 1
            avg_time_per_step = elapsed / epochs_done
            eta = avg_time_per_step * (args.adapt_epochs - epochs_done)
            print(f"\r[Epoch {epoch}/{args.adapt_epochs}] "f"Loss = {avg_loss:.4f} | ETA: {int(eta)}s", end="")
        print()
        if args.init_params in ['ta','tsv','wudi']:
            merged_model.model.visual.transformer.resblocks[layer_idx] = merged_layer.get_merged_model().cpu()
        else:
            merged_model.model.visual.transformer.resblocks[layer_idx] = merged_layer.cpu()
    return merged_model


if __name__ == '__main__':
    if args.name=='init_by_weights':
        args.read_path=f'{args.read_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-1.0-opt-{args.opt_steps}-{args.opt_lr}'
        save_path = f'{args.model_location}/{args.model}/{args.name}/anchor-{args.anchor_num}-scale-1.0-adapt-{args.adapt_epochs}-{args.adapt_lr}-{args.adapt_batch_size}-{args.adapt_loss}'
    elif args.name=='init_by_gauss_random':
        args.read_path=f'{args.read_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-{args.scale}-opt-{args.opt_steps}-{args.opt_lr}'
        save_path = f'{args.model_location}/{args.model}/{args.name}/anchor-{args.anchor_num}-scale-{args.scale}-adapt-{args.adapt_epochs}-{args.adapt_lr}-{args.adapt_batch_size}-{args.adapt_loss}'
    else:
        raise ValueError(f'Invalid name: {args.name}')
    save_hyperparams(args, save_path)
    merged_model = train(args)
    
    os.makedirs(save_path, exist_ok=True)
    torch.save(merged_model, save_path+f'/merged_model_by_{args.init_params}.pt')

    metrics = {}
    dataset_list =['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB','MNIST','DTD'] 
    for dataset in dataset_list:
        metrics[dataset] = eval_single_dataset(merged_model, dataset + 'ValfromTrain', args, dataset, None, None)['top1']
    metrics['avg'] = sum(metrics.values()) / len(metrics)
    print(metrics)
                        
    metrics_file = os.path.join(save_path, f'results_by_{args.init_params}.txt')
    with open(metrics_file, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")