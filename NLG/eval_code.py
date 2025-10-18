import json
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import sys
import tqdm
import io
import signal
import glob
from vllm import SamplingParams
import jsonlines
from utils.evaluate_llms_utils import *
from typing import Iterable, Dict
import logging
import torch
from model_merging_methods.distill_merging_utils import *

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def eval_code(llm, args):
    def handler(signum, frame):
        raise TimeoutError("TLE")

    def capture_output_with_timeout(code_string, timeout=2):
        signal.signal(signal.SIGALRM, handler)
        output_capture = io.StringIO()
        
        sys.stdout = output_capture
        
        try:
            signal.alarm(timeout)
            exec(code_string)
            signal.alarm(0)
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)
        finally:
            signal.alarm(0)
            sys.stdout = sys.__stdout__
        
        output = output_capture.getvalue()
        
        return output, None

    os.environ["WANDB_DISABLED"] = "true"

    def test_human_eval(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
        problems = read_mbpp(test_data_path)
        task_ids = sorted(problems.keys())[start_index: end_index]
        prompts = [problems[task_id]['prompt'] for task_id in task_ids]
        num_samples = len(prompts)
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

        os.makedirs(save_gen_results_folder, exist_ok=True)

        batch_size = 64
        for i in tqdm.tqdm(range(0, num_samples, batch_size)):
            batch_prompts = prompts[i: i + batch_size]
            for j in range(len(batch_prompts)):
                batch_prompts[j] = batch_prompts[j].replace('    ', '\t')
                batch_prompts[j] = generate_code_task_prompt(batch_prompts[j])
            batch_task_ids = task_ids[i: i + batch_size]
            batch_completion_seqs = []

            with torch.no_grad():
                completions = llm.generate(batch_prompts, sampling_params)
            gen_seqs = [completion.outputs[0].text for completion in completions]

            for j, gen_seq in enumerate(gen_seqs):
                completion_seq = gen_seq.split("### Response:")[-1]
                completion_seq = completion_seq.replace('\t', '    ')
                all_code = gen_seq.replace('\t', '    ')

                batch_completion_seqs.append(
                    {'task_id': batch_task_ids[j],
                        'completion': completion_seq,
                        'all_code': all_code,
                        }
                )

            output_file = f"{save_gen_results_folder}/{i}.jsonl"
            print(f"save to {output_file}")
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(batch_completion_seqs)

        files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
        print(f"find {len(files)} files in {save_gen_results_folder}")

        outputs = []
        for code_file in tqdm.tqdm(files, total=len(files)):
            codes = [c for c in stream_jsonl(code_file)]
            for code in codes:
                completion = code['completion']
                completion = completion.replace("\r", "")
                completion = completion.strip()
                if '```python' in completion:
                    # print("completion matches ```python")
                    def_line = completion.index('```python')
                    completion = completion[def_line:].strip()
                    completion = completion.replace('```python', '')
                    try:
                        next_line = completion.index('```')
                        completion = completion[:next_line].strip()
                    except:
                        # print("wrong completion")
                        a=0
                if "__name__ == \"__main__\"" in completion:
                    # print("completion matches __name__ == \"__main__\"")
                    try:
                        next_line = completion.index('if __name__ == "__main__":')
                        completion = completion[:next_line].strip()
                    except:
                        # print("wrong completion")
                        a=0
                if "# Example usage" in completion:
                    # print("completion matches # Example usage")
                    next_line = completion.index('# Example usage')
                    completion = completion[:next_line].strip()
                # the following codes are used to deal with the outputs of code-alpaca
                if "The solution is:" in completion:
                    # print("completion matches The solution is:")
                    def_line = completion.index("The solution is:")
                    completion = completion[def_line:].strip()
                    completion = completion.replace('The solution is:', '')
                    try:
                        next_line = completion.index('\n\nThe answer is:')
                        completion = completion[:next_line].strip()
                    except:
                        completion = completion.strip()
                        # print("maybe wrong completion")
                if "The answer is:" in completion:
                    # print("completion matches The answer is:")
                    def_line = completion.index("The answer is:")
                    completion = completion[def_line:].strip()
                    completion = completion.replace('The answer is:', '')
                    try:
                        next_line = completion.index('\n\nThe answer is:')
                        completion = completion[:next_line].strip()
                    except:
                        completion = completion.strip()
                        # print("maybe wrong completion")
                code = problems[code['task_id']]['prompt'] + completion + "\n" + problems[code['task_id']]['test'] + "\n" + f"check({problems[code['task_id']]['entry_point']})"
                outputs.append(code)

        print(f"save to {save_gen_results_folder}.jsonl")
        with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
            json.dump(outputs, fout)


    def test_mbpp(llm, test_data_path, args, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
        try:
            code_indices = torch.load(f'{args.model_path}/code_indices.pt')
            code_eval_indices = {}
            for idx, i in enumerate(code_indices):
                code_eval_indices[i] = idx
        except:
            code_eval_indices = {}
            for i in range(10000):
                code_eval_indices[i] = i
        problems = read_mbpp(test_data_path)
        task_ids = sorted(problems.keys())[start_index: end_index]
        prompts = []
        for task_id in task_ids:
            idx = task_id - 11
            if idx not in code_eval_indices:
                # pop the task_id that is not in the code_eval_indices
                task_ids.pop(task_ids.index(task_id))
        for task_id in task_ids:
            prompt = f"\n{problems[task_id]['text']}\nTest examples:"
            if task_id == 493:
                # The test examples are too long, we choose to only include the function name.
                test_example = problems[task_id]['test_list'][0]
                prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
            else:
                for test_example in problems[task_id]['test_list']:
                    prompt += f"\n{test_example}"
            prompts.append(prompt)

        num_samples = len(prompts)
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

        os.makedirs(save_gen_results_folder, exist_ok=True)

        batch_size = 64
        for i in tqdm.tqdm(range(0, num_samples, batch_size)):
            batch_prompts = prompts[i: i + batch_size]
            for j in range(len(batch_prompts)):
                batch_prompts[j] = batch_prompts[j].replace('    ', '\t')
                batch_prompts[j] = generate_code_task_prompt(batch_prompts[j])
            batch_task_ids = task_ids[i: i + batch_size]
            batch_completion_seqs = []

            with torch.no_grad():
                completions = llm.generate(batch_prompts, sampling_params)
            gen_seqs = [completion.outputs[0].text for completion in completions]

            for j, gen_seq in enumerate(gen_seqs):
                completion_seq = gen_seq.split("### Response:")[-1]
                completion_seq = completion_seq.replace('\t', '    ')
                all_code = gen_seq.replace('\t', '    ')

                batch_completion_seqs.append(
                    {'task_id': batch_task_ids[j],
                        'completion': completion_seq,
                        'all_code': all_code,
                        }
                )

            output_file = f"{save_gen_results_folder}/{i}.jsonl"
            print(f"save to {output_file}")
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(batch_completion_seqs)

        files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
        print(f"find {len(files)} files in {save_gen_results_folder}")

        problems = read_mbpp(test_data_path)
        outputs = []

        for code_file in tqdm.tqdm(files, total=len(files)):
            codes = [c for c in stream_jsonl(code_file)]
            for code in codes:
                task_id = code['task_id']
                completion = code['completion']
                completion = completion.strip()
                if '```python' in completion:
                    # print("completion matches ```python")
                    def_line = completion.index('```python')
                    completion = completion[def_line:].strip()
                    completion = completion.replace('```python', '')
                    try:
                        next_line = completion.index('\n```')
                        completion = completion[:next_line].strip()
                    except:
                        a=0
                        # print("wrong completion")
                if "__name__ == \"__main__\"" in completion:
                    print("completion matches __name__ == \"__main__\"")
                    try:
                        next_line = completion.index('if __name__ == "__main__":')
                        completion = completion[:next_line].strip()
                    except:
                        a=0
                        # print("wrong completion")
                if "# Example usage" in completion:
                    # print("completion matches # Example usage")
                    next_line = completion.index('# Example usage')
                    completion = completion[:next_line].strip()
                if "# Test examples" in completion:
                    # print("completion matches # Test examples")
                    next_line = completion.index('# Test examples')
                    completion = completion[:next_line].strip()
                # the following codes are used to deal with the outputs of code-alpaca
                if "The solution is:" in completion:
                    # print("completion matches The solution is:")
                    def_line = completion.index("The solution is:")
                    completion = completion[def_line:].strip()
                    completion = completion.replace('The solution is:', '')
                    try:
                        next_line = completion.index('\n\nThe answer is:')
                        completion = completion[:next_line].strip()
                    except:
                        completion = completion.strip()
                        # print("maybe wrong completion")
                if "The answer is:" in completion:
                    # print("completion matches The answer is:")
                    def_line = completion.index("The answer is:")
                    completion = completion[def_line:].strip()
                    completion = completion.replace('The answer is:', '')
                    try:
                        next_line = completion.index('\n\nThe answer is:')
                        completion = completion[:next_line].strip()
                    except:
                        completion = completion.strip()
                        # print("maybe wrong completion")
                
                completion = completion
                for test_example in problems[task_id]['test_list']:
                    completion += f"\n{test_example}"
                outputs.append(completion)

        print(f"save to {save_gen_results_folder}.jsonl")
        with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
            json.dump(outputs, fout)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"test.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    if args.dataset == "mbpp":
        test_mbpp(llm, "math_code_data/mbpp.test.jsonl", args, logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=f"mbpp_test_results/{args.model_path}")
        acc = 0
        total = 0
        with open(f"mbpp_test_results/{args.model_path}.jsonl", "r") as f:
            # content is a listz
            content = json.load(f)
        print(len(content))
        for i, item in enumerate(content):
            output, error = capture_output_with_timeout(item, timeout=60)
            if error is None:
                acc += 1
            total += 1
        logger.info(f"{args.model_path} MBPP Accuracy: {acc / total}")

    elif args.dataset == "human_eval":
        test_human_eval(llm, "math_code_data/human_eval.jsonl", args, logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=f"human_eval_results/{args.model_path}")

        acc = 0
        total = 0
        with open(f"human_eval_results/{args.model_path}.jsonl", "r") as f:
            # content is a list
            content = json.load(f)
        
        for i, item in enumerate(content):
            output, error = capture_output_with_timeout(item, timeout=60)
            if error is None:
                acc += 1
            total += 1
        
        logger.info(f"{args.model_path} Human Eval Accuracy: {acc / total}")