import json
import os
import os.path as osp
import argparse
import multiprocessing as mp
import time
import logging
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from tqdm import tqdm
from logging import Logger
logger = Logger("ablation-data-comp")


def setup_logger(log_file=None):
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    ch.setLevel("INFO")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # file
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel("INFO")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_zh_dataset_jsonl", default="/workspace/processed_data/zh_dataset.jsonl"
    )
    parser.add_argument(
        "--input_en_dataset_jsonl", default="/workspace/processed_data/en_dataset.jsonl"
    )
    parser.add_argument("--remove_zh_keys", nargs="+", default=[])
    parser.add_argument("--remove_en_keys", nargs="+", default=[])
    parser.add_argument(
        "--exp_root_dir", default="/workspace/my_scripts/ablation-data-comp"
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20000317])
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--eq_prob_split", action="store_true")
    args = parser.parse_args()
    # assert len(args.gpus) == 4, "Only support 4 GPUs!"
    setup_logger(osp.join(args.exp_root_dir, "logs", args.exp_name + "-full-log.log"))

    for seed in args.seeds:
        args.seed = seed
        run(args)


def process_eq_split(dataset: Dataset, seed: int):
    path_nums = {}
    for item in tqdm(dataset, ascii=False):
        path = item["meta"]["original_path"]
        if path not in path_nums:
            path_nums[path] = 0
        path_nums[path] += 1
    min_num = min(path_nums.values())
    min_num = max(20000, min_num)
    logger.info("Min: {}".format(min_num))
    new_dataset = []
    for key in path_nums:
        curr_dataset = dataset.filter(
            lambda item: item["meta"]["original_path"] == key
        ).shuffle(seed)
        if len(curr_dataset) > min_num:
            curr_dataset = curr_dataset.select(range(min_num))
        new_dataset.append(curr_dataset)
    new_dataset = concatenate_datasets(new_dataset)
    return new_dataset


def run(args):
    EXP_PATH = osp.join(args.exp_root_dir, args.exp_name + "_seed_" + str(args.seed))
    zh_dataset_name = "zh_dataset.jsonl"
    en_dataset_name = "en_dataset.jsonl"
    os.makedirs(EXP_PATH, exist_ok=(args.exp_name == "debug"))

    # Filtering
    zh_dataset = load_dataset(
        "json", data_files=args.input_zh_dataset_jsonl, split="train"
    )
    en_dataset = load_dataset(
        "json", data_files=args.input_en_dataset_jsonl, split="train"
    )
    print(
        "Original: zh_dataset: {}, en_dataset: {}".format(
            len(zh_dataset), len(en_dataset)
        )
    )
    if (len(args.remove_zh_keys + args.remove_en_keys) == 0):
        print("Skip filtering.")
        exit_code_1 = os.system("ln -s {} {}".format(
            args.input_zh_dataset_jsonl,
            osp.join(EXP_PATH, "zh_dataset.jsonl"),
        ))
        exit_code_2 = os.system("ln -s {} {}".format(
            args.input_en_dataset_jsonl,
            osp.join(EXP_PATH, "en_dataset.jsonl"),
        ))
        assert not (exit_code_1 or exit_code_2), "Symlink failed!"
    else:
        zh_dataset = zh_dataset.filter(
            lambda item: item["meta"]["original_path"] not in args.remove_zh_keys
        )
        en_dataset = en_dataset.filter(
            lambda item: item["meta"]["original_path"] not in args.remove_en_keys
        )
        print(
            "Filtered: zh_dataset: {}, en_dataset: {}".format(
                len(zh_dataset), len(en_dataset)
            )
        )
        zh_dataset.to_json(osp.join(EXP_PATH, "zh_dataset.jsonl"), force_ascii=False)
        en_dataset.to_json(osp.join(EXP_PATH, "en_dataset.jsonl"), force_ascii=False)
    if args.eq_prob_split:
        zh_dataset = process_eq_split(zh_dataset, args.seed)
        en_dataset = process_eq_split(en_dataset, args.seed)
        zh_dataset.to_json(osp.join(EXP_PATH, "zh_dataset_presampled.jsonl"), force_ascii=False)
        en_dataset.to_json(osp.join(EXP_PATH, "en_dataset_presampled.jsonl"), force_ascii=False)
        zh_dataset_name = "zh_dataset_presampled.jsonl"
        en_dataset_name = "en_dataset_presampled.jsonl"

    # Sampling
    cmd = "python {} --en_data_dir {} --zh_data_dir {} --output_files {} --my_seed {}".format(
        "/workspace/FT-Data-Ranker-7B/get_train_dataset_7b_seed.py",
        osp.join(EXP_PATH, en_dataset_name),
        osp.join(EXP_PATH, zh_dataset_name),
        osp.join(EXP_PATH, "data_7b.jsonl"),
        args.seed
    )
    print("Sampling command: {}".format(cmd))
    exit_code = os.system(cmd)
    assert exit_code == 0, "Sampling failed!"

    # SFT
    cmd = (
        "cd /workspace/lm-trianing; "
        "bash /workspace/lm-trianing/train_scripts/deepspeed_train_7b_lora.sh {} {} {} {} {} {}".format(
            "/official-datasets-and-models/data/models/Baichuan2-7B-Base/",
            osp.join(EXP_PATH, "data_7b.jsonl"),
            osp.join(EXP_PATH, "7b_lora_model"),
            ",".join(list(map(str, args.gpus))),
            args.port,
            len(args.gpus),
        )
    )
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    print("SFT command: {}".format(cmd))
    exit_code = os.system(cmd)
    assert exit_code == 0, "SFT failed!"
    cmd = "python /workspace/FT-Data-Ranker-7B/transform_to_fp16.py {} {}".format(
        osp.join(EXP_PATH, "7b_lora_model"), osp.join(EXP_PATH, "7b_lora_model_fp16")
    )
    exit_code = os.system(cmd)
    assert exit_code == 0, "FP16 conversion failed!"

    # Evaluate
    cmd_dev = "cd /workspace/lm-evaluation-harness; bash examples/challenge-7B-stage1.sh {} {} {} {}".format(
        "dev",
        osp.join(EXP_PATH, "7b_lora_model_fp16"),
        "/official-datasets-and-models/data/challenge-data",
        osp.join(EXP_PATH, "eval_results"),
    )
    with open(osp.join(EXP_PATH, "cmd_dev.sh"), "w") as f:
        print(cmd_dev, file=f)
    cmd_board = "cd /workspace/lm-evaluation-harness; bash examples/challenge-7B-stage1.sh {} {} {} {}".format(
        "board",
        osp.join(EXP_PATH, "7b_lora_model_fp16"),
        "/official-datasets-and-models/data/challenge-data",
        osp.join(EXP_PATH, "eval_results"),
    )
    with open(osp.join(EXP_PATH, "cmd_board.sh"), "w") as f:
        print(cmd_board, file=f)
    nohup_cmd_dev = (
        "export CUDA_VISIBLE_DEVICES={}; nohup {} > {}/dev.log 2>&1 &".format(
            args.gpus[0], "bash {}".format(osp.join(EXP_PATH, "cmd_dev.sh")), EXP_PATH
        )
    )
    nohup_cmd_board = (
        "export CUDA_VISIBLE_DEVICES={}; nohup {} > {}/board.log 2>&1 &".format(
            args.gpus[1], "bash {}".format(osp.join(EXP_PATH, "cmd_board.sh")), EXP_PATH
        )
    )
    print("Dev command: {}".format(nohup_cmd_dev))
    print("Board command: {}".format(nohup_cmd_board))
    os.system(nohup_cmd_dev)
    os.system(nohup_cmd_board)
    while True:
        with open(osp.join(EXP_PATH, "dev.log"), "r") as f:
            dev_log = f.read()
        with open(osp.join(EXP_PATH, "board.log"), "r") as f:
            board_log = f.read()
        if not ("[Done]" in dev_log and "[Done]" in board_log):
            time.sleep(60)
            continue
        break
    os.system(
        "python /workspace/my_scripts/read_eval_res.py {} > {}".format(
            osp.join(EXP_PATH, "eval_results", "dev"),
            osp.join(EXP_PATH, "dev_res.json"),
        )
    )


if __name__ == "__main__":
    main()
