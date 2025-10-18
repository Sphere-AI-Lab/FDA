import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import sys
from vllm import LLM
import argparse
from utils.evaluate_llms_utils import *
import torch
from model_merging_methods.distill_merging_utils import *
import eval_math, eval_code

cache_dir = "./MergeLM_models"

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--do_instruct", action="store_true", help="whether to merge instruct model")
parser.add_argument("--do_math", action="store_true", help="whether to merge math model")
parser.add_argument("--do_code", action="store_true", help="whether to merge code model")
parser.add_argument("--language_model_name", type=str,
                    default="Llama-2-13b-hf_32001", help="name of the language model")
parser.add_argument("--merging_method_name", type=str,
                    default="sequential_efficient")
parser.add_argument("--val_shot", type=int, default=64,
                    help="number of examples sampled from training set for validation")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")

parser.add_argument("--tag", type=str,
                    default='test', help="tag for distill merging")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--layer_save", type=str, default="./save_layers", help="path to save layers in merging")
parser.add_argument("--model_path", type=str, default="save_merge_models/all/sequential_efficient/Llama-2-13b-hf_32001/64")
parser.add_argument("--dataset", type=str, default="math")
try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available(
    ) and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()

args.dataset_names = ["math", "code"]
args.cache_dir = cache_dir



if __name__ == "__main__":
    llm = LLM(args.model_path, tensor_parallel_size=args.gpu, trust_remote_code=True)

    if args.do_instruct:
        eval_instruct.eval_instruct(llm, args)

    if args.do_code:
        args.dataset = "mbpp"
        eval_code.eval_code(llm, args)
        args.dataset = "human_eval"
        eval_code.eval_code(llm, args)

    if args.do_math:
        args.dataset = "gsm8k"
        eval_math.eval_math(llm, args)
        args.dataset = "math"
        eval_math.eval_math(llm, args)
