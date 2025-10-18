import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from utils.glue_data_loader import GLUEDataLoader,glue_data_metrics_map
from utils.customized_trainers import CustomizedTrainer
from utils.metrics import compute_metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import torch
from functools import partial
from fda_args import parse_arguments
import os
import torch.nn.functional as F
import time
import numpy as np

from model_merging_methods.distill_merging_utils import *
from datasets import logging as datasets_logging
datasets_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 禁用高效注意力
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


args = parse_arguments()

seed_torch(args.seed) 

# save hyperparams
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
    if args.model=="roberta-large":
        num_layers=24
        args.alpha=0.2 # here, the alpha is chosed based on the TA performance. So different ckpts leads to different alpha.
    else:
        num_layers=12
        args.alpha=0.3

    print(args.init_params)
    merged_model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=f'{args.model_location}/{args.model}').to(args.device)
    for layer_idx in range(num_layers):
        print('\nStage1: Read saved anchors for layer ',layer_idx)
        X_dict = {}
        for dataset in args.dataset_names:
            read_path=f"{args.read_path}/{dataset}"
            read_file=os.path.join(read_path,f'dual_anchors_layer_{layer_idx+1}.npy')
            anchors = np.load(read_file,allow_pickle=True).item()
            anchors = {key: torch.from_numpy(value).float().cuda() for key, value in anchors.items()} 
            X_dict[dataset] = anchors
        
        print('Stage 2: Aligning the layer ',layer_idx)
        dataset = LabeledMultiDatasetFromDict(X_dict)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.adapt_batch_size, shuffle=True)
        if args.init_params=='pretrained':
            merged_layer=merged_model.roberta.encoder.layer[layer_idx]
            # As in previous experiemnts, we don't use the train mode for adapte the model initialized from pretrained model.
            # Thus, we keep this settings. The hyperparameter settings in the train mode are encouraged to explore.
        elif args.init_params in ['ta','tsv','wudi']:
            merged_layer = load_merged_layers(args, layer_idx=layer_idx)
            merged_layer.train()
        else:
            raise ValueError(f'Invalid init_params: {args.init_params}')
        optimizer = torch.optim.Adam(merged_layer.parameters(), lr=args.adapt_lr)
        start_time=time.time()
        for epoch in range(args.adapt_epochs):
            epoch_loss = 0
            batch_count = 0
            for x,y,mask in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                mask=mask.to(args.device)
                optimizer.zero_grad()
                if args.init_params in ['ta','tsv','wudi']:
                    out_merged=merged_layer.get_merged_model()(x,mask, None, None, None, None, False)[0].reshape(x.shape[0], -1)
                elif args.init_params == 'pretrained':
                    out_merged=merged_layer(x,mask, None, None, None, None, False)[0].reshape(x.shape[0], -1)
                y=y.reshape(y.shape[0],-1)
                if args.adapt_loss == 'cos':
                    loss = -F.cosine_similarity(out_merged, y, dim=1).sum()
                elif args.adapt_loss == 'mse':
                    loss = F.mse_loss(out_merged, y, reduction='sum')
                elif args.adapt_loss == 'l1':
                    loss = F.l1_loss(out_merged, y, reduction='sum')
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
            merged_model.roberta.encoder.layer[layer_idx] = merged_layer.get_merged_model()
        else:
            merged_model.roberta.encoder.layer[layer_idx] = merged_layer
        del merged_layer, dataset, loader
        torch.cuda.empty_cache()
    return merged_model


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

    load_model_paths = []
    for dataset_name in args.dataset_names:
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.model}"]
        load_model_paths.append(f"{args.model_location}/{dataset_name}/{args.model}_lr{learning_rate}")
    
    args.load_model_paths_dict = {args.dataset_names[i]: load_model_paths[i] for i in range(len(args.dataset_names))}

    if 'weights' in args.name:
        args.read_path=f'{args.read_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-1.0-opt-{args.opt_steps}-{args.opt_lr}'
        save_path = f'{args.model_location}/{args.model}/{args.name}/anchor-{args.anchor_num}-scale-1.0-adapt-{args.adapt_epochs}-{args.adapt_lr}-{args.adapt_batch_size}-{args.adapt_loss}'
    elif 'random' in args.name:
        args.read_path=f'{args.read_anchors_path}/{args.model}/{args.name}-anchor-{args.anchor_num}-scale-{args.scale}-opt-{args.opt_steps}-{args.opt_lr}'
        save_path = f'{args.model_location}/{args.model}/{args.name}/anchor-{args.anchor_num}-scale-{args.scale}-adapt-{args.adapt_epochs}-{args.adapt_lr}-{args.adapt_batch_size}-{args.adapt_loss}'
    else:
        raise ValueError(f'Invalid name: {args.name}')

    save_hyperparams(args, save_path)
    merged_model = train(args)
    os.makedirs(save_path, exist_ok=True)
    torch.save(merged_model, save_path+f'/merged_model_by_{args.init_params}.pt')

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

    # Save the results to a file
    metrics_file = os.path.join(save_path, f'results_by_{args.init_params}.txt')
    with open(metrics_file, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"Results saved to {metrics_file}")
    