import torch
from transformers import AutoModelForCausalLM
import json
import os
import argparse

# Parse arguments /home/shikexuan/prodistill
parser = argparse.ArgumentParser()
parser.add_argument('--llm_version', type=str, default='v1.0')
args = parser.parse_args()

# Load the full model
model_path="/data/shikexuan/save_merged_models_dare/math_code/code/task_arithmetic_scaling_coefficient_0.5"
split_dir = os.path.join(model_path, f'split')
os.makedirs(split_dir, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Prepare the weight map
weight_map = {}

# Save embedding layer
torch.save(model.model.embed_tokens, os.path.join(split_dir, 'model_embed_tokens.pt'))
weight_map['model.embed_tokens'] = 'model_embed_tokens.pt'

# Iterate through layers to save individual layers directly
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    
    # Save each layer directly
    torch.save(layer, os.path.join(split_dir, f'model_layer_{i}.pt'))
    weight_map[f'model.layers.{i}'] = f'model_layer_{i}.pt'

# Save normalization and lm_head
torch.save(model.model.norm, os.path.join(split_dir, 'model_norm.pt'))
weight_map['model.norm'] = 'model_norm.pt'

# torch.save(model.model.rotary_emb, os.path.join(split_dir, 'model_rotary_emb.pt'))
# weight_map['model.rotary_emb'] = 'model_rotary_emb.pt'

torch.save(model.lm_head, os.path.join(split_dir, 'model_lm_head.pt'))
weight_map['lm_head'] = 'model_lm_head.pt'

# Save the index file with all mappings
with open(os.path.join(split_dir, 'model_index.json'), 'w') as index_file:
    json.dump(weight_map, index_file, indent=4)

print('Model structure and weights have been saved, and the index file has been generated.')
