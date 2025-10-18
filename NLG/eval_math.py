import json
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import sys
import io
import signal
from vllm import SamplingParams
import jsonlines
from utils.evaluate_llms_utils import *
import logging
import torch
from model_merging_methods.distill_merging_utils import *


def eval_math(llm, args):
    os.environ["WANDB_DISABLED"] = "true"

    if args.dataset == 'gsm8k':
        data_path = 'math_code_data/gsm8k_test.jsonl'
        try:
            math_indices = torch.load(f'{args.model_path}/math_indices.pt')
            math_eval_indices = {}
            for idx, i in enumerate(math_indices):
                math_eval_indices[i] = idx
        except:
            math_eval_indices = {}
            for i in range(10000):
                math_eval_indices[i] = i
    elif args.dataset == 'math':
        data_path = 'math_code_data/MATH_test.jsonl'
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    questions = []
    problem_prompt = get_math_task_prompt()
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"test.log"),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger()
    logger.info(f"Model path is {args.model_path}")
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if args.dataset == 'gsm8k':
                if idx not in math_eval_indices:
                    print(idx)
                    continue
            if args.dataset == 'gsm8k':
                question = item["question"]
            elif args.dataset == 'math':
                question = item["instruction"]
            questions.append(question)
            prompt = problem_prompt.format(instruction=question)
            hendrycks_math_ins.append(prompt)
            if args.dataset == 'gsm8k':
                solution = item["answer"]
                temp_ans = solution.split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))
            elif args.dataset == 'math':
                solution = item["output"]
                temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=64)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    if args.dataset=='gsm8k':
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop=stop_tokens) # 这样子，就是0.5888
    else:
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens) # 这样子，就是0.5888
    # sampling_params = SamplingParams(temperature=0.9, top_p=0.6, max_tokens=2048, stop=stop_tokens) # 这样子，可以到0.60

    res_completions = []
    total = 0
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            total += 1
            res_completions.append(generated_text)
    
    print(len(res_completions), len(hendrycks_math_ins), len(hendrycks_math_answers))

    results = []
    invalid_outputs = []

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
            sys.stdout = sys.__stdout__
        
        output = output_capture.getvalue()
        
        return output, None
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        # print(completion)
        if 'Qwen' in args.model_path:
            answer = completion.split('\\boxed{')[-1].split('}')[0]
            if answer != None:
                try :
                    results.append(float(answer) == float(prompt_answer))
                except:
                    results.append(str(answer) == str(prompt_answer))
            else:
                results.append(False)
        elif args.dataset == 'gsm8k':
            answer = extract_answer_number(completion)
            if answer != None:
                try :
                    if float(answer) == float(prompt_answer) or math_equal(answer, prompt_answer):
                        results.append(True)
                    else:
                        results.append(False)
                except:
                    if str(answer) == str(prompt_answer) or math_equal(answer, prompt_answer):
                        results.append(True)
                    else:
                        results.append(False)
            else:
                results.append(False)
        elif args.dataset == 'math':
            res = process_results(prompt, completion, prompt_answer, invalid_outputs)
            results.append(res)
    accuracy = sum(results) / len(results)

    logger.info(f"{args.model_path} {args.dataset} Accuracy is {accuracy}")