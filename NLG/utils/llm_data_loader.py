import numpy as np
import json
import pandas as pd

# os.environ['HF_DATASETS_OFFLINE'] = '1'

from torch.utils.data import Subset, Dataset
import datasets
import transformers
from utils.evaluate_llms_utils import *

class LLMDataLoader:
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.tokenizer = tokenizer
        self.alpaca_path = "math_code_data/alpaca_eval.json"
        self.math_path = "math_code_data/gsm8k_test.jsonl"
        self.mbpp_path = "math_code_data/mbpp.test.jsonl"
        self.max_len = 0

    def encode(self, examples: dict, max_seq_length: int = 512):
        inputs = {}
        ins_token = self.tokenizer(examples['instruction'], max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        inputs['input_ids'] = ins_token['input_ids']
        inputs['attention_mask'] = ins_token['attention_mask']
        self.max_len = max(self.max_len, inputs['attention_mask'][0].sum().item())
        # target = self.tokenizer(examples['output'], max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
        # inputs['labels'] = target['input_ids']
        # self.max_len = max(self.max_len, target['attention_mask'][0].sum().item())        
        return inputs
        
    def load_dataset(self, dataset_name: str, max_seq_length: int = 256, val_shot: int = 64):
        # train: 64, other is test
        if dataset_name == "instruct":
            with open(self.alpaca_path, 'r') as f:
                alpaca_data = json.load(f)
                data_df = pd.DataFrame(alpaca_data)[['instruction', 'output']]
            # use generate_instruction_following_task_prompt
            data_df['instruction'] = data_df['instruction'].apply(lambda x: generate_instruction_following_task_prompt(x, is_chat_model=True))
        elif dataset_name == "math":
            math_data = pd.read_json(self.math_path, lines=True)
            data_df = math_data[['question', 'answer']]
            data_df = data_df.rename(columns={"question": "instruction", "answer": "output"})
            # use get_math_task_prompt
            data_df['instruction'] = data_df['instruction'].apply(lambda x: get_math_task_prompt().format(instruction=x))
        elif dataset_name == "code":
            mbpp_data = pd.read_json(self.mbpp_path, lines=True)
            data_df = mbpp_data[['text', 'code']]
            data_df = data_df.rename(columns={"text": "instruction", "code": "output"})
            # use generate_code_task_prompt
            data_df['instruction'] = data_df['instruction'].apply(generate_code_task_prompt)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        dataset = datasets.Dataset.from_pandas(data_df)
        dataset = dataset.map(self.encode, batched=True)

        permuted_indices = np.random.RandomState(seed=0).permutation(len(dataset)).tolist()
        num_train_data = val_shot
        train_dataset = Subset(dataset=dataset, indices=permuted_indices[:num_train_data])
        test_dataset = Subset(dataset=dataset, indices=permuted_indices[num_train_data:])
        return train_dataset, test_dataset

