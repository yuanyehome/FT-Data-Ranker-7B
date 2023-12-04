import os
import argparse
import random
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=float, default=0.8)
parser.add_argument("--en_data_dir",
                    default="/workspace/processed_data/en_dataset.jsonl")
parser.add_argument("--zh_data_dir",
                    default="/workspace/processed_data/zh_dataset.jsonl")
parser.add_argument("--output_files",
                    default="/workspace/processed_data/example_data_7b.jsonl")
parser.add_argument("--my_seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.my_seed)
TOKEN_NUMS = 10000000
RATIO = args.ratio
EN_DATA_DIR  = args.en_data_dir
ZH_DATA_DIR  = args.zh_data_dir
OUTPUT_FILES = args.output_files

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Input:\n\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Response:"
    ),
}

def get_token_count(sample):
    instruction_len = len(enc.tokenize(sample['instruction']))
    input_len = len(enc.tokenize(sample['input']))
    output_len = len(enc.tokenize(sample['output']))
    total_prompt_input_len = instruction_len + input_len + prompt_input_len
    return total_prompt_input_len

def findAllFile(base):
    files = []
    if os.path.isfile(base):
        return [base]
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('jsonl'):
                files.append(os.path.join(root, f))
    return files

enc = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
enc.model_max_length = 1000000000000000019884624838656
prompt_input_len = len(enc.tokenize(PROMPT_DICT["prompt_input"]))

en_files = findAllFile(EN_DATA_DIR)
ds_en = load_dataset('json', data_files=en_files, split='train').shuffle(seed=args.my_seed)
en_token_nums = TOKEN_NUMS * RATIO if ZH_DATA_DIR else TOKEN_NUMS
zh_token_nums = TOKEN_NUMS - en_token_nums

count = 0
selected_idxs = []
for i in range(len(ds_en)):
    selected_idxs.append(i)
    count += get_token_count(ds_en[i])
    print('en num_tokens', i, count)
    if count >= en_token_nums:
        break
ds_en = ds_en.select(selected_idxs).select_columns(['instruction', 'input', 'output'])

if ZH_DATA_DIR:
    zh_files = findAllFile(ZH_DATA_DIR)
    ds_zh = load_dataset('json', data_files=zh_files, split='train').shuffle(seed=args.my_seed)
    count = 0
    selected_idxs = []
    for i in range(len(ds_zh)):
        selected_idxs.append(i)
        count += get_token_count(ds_zh[i])
        print('zh num_tokens', i, count)
        if count >= zh_token_nums:
            break
    ds_zh = ds_zh.select(selected_idxs).select_columns(['instruction', 'input', 'output'])
    ds = concatenate_datasets([ds_en, ds_zh]).shuffle(seed=args.my_seed)
    ds.to_json(OUTPUT_FILES, force_ascii=False)
else:
    ds_en.to_json(OUTPUT_FILES, force_ascii=False)
