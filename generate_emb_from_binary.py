#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
from cptools import LogHandler, read_pickle, read_json, write_json
from data_extraction.extract_functions import extract_functions_from_binary
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pickle

from model_utils import BertCustomModel
from data_utils import DataCollatorFinetuneSimcseForLanguageModelingEval, DataCollatorFinetuneSimcseForLanguageModeling

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
import torch
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools
import time

BATCH_SIZE = 128


def get_arch_id(opt):

    if opt == 'x86':
        return 0
    if opt == 'arm':
        return 1
    if opt == 'mips':
        return 2
    return -1


def gen_hex_embedding(device, data_list, batch_size, tokenizer_name, config_name, model_name_or_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False,do_lower_case=False, do_basic_tokenize=False)

    config = AutoConfig.from_pretrained(config_name)
    
    model = BertCustomModel.from_pretrained(
            model_name_or_path, 
            config=config,
            add_pooling_layer=True,
            custom_config={'arch_num':3}
    )
    
    max_seq_length = 512

    def tokenize_function(examples):
        src_result = tokenizer(
            examples['hex'],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        src_result['arch_ids'] = [[i] * max_seq_length for i in examples['arch_id']]

        return src_result
    
    import datasets
    batch_dataset = datasets.Dataset.from_list(data_list)

    column_names = batch_dataset.column_names
    tokenized_datasets = batch_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        desc="Running tokenizer on dataset line_by_line",
    )

    batch_tok_dataset = tokenized_datasets

    data_collator = DataCollatorFinetuneSimcseForLanguageModelingEval(tokenizer=tokenizer)

    anchor_dataloader = DataLoader(batch_tok_dataset, collate_fn=data_collator, batch_size=batch_size)

    start = time.time()

    model.to(device)
    model.eval()
    batch_all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(anchor_dataloader, desc="anchor_batch"):
            input_ids = batch['input_ids']
            token_type_ids = batch['token_type_ids']
            attention_mask = batch['attention_mask']
            arch_ids = batch['arch_ids']

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            arch_ids = arch_ids.to(device)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, arch_ids=arch_ids)
            embeddings = outputs.pooler_output.detach().cpu()
            batch_all_embeddings.append(embeddings)
    anchor_final_embeddings = torch.cat(batch_all_embeddings, dim=0)

    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")

    return anchor_final_embeddings.numpy()


def main(args):

    tokenizer_name = args.tokenizer_name
    config_name = args.config_name
    model_name_or_path = args.model_path

    device = torch.device('cpu')
    if args.is_gpu:
        device = torch.device('cuda')

    func_dict, arch = extract_functions_from_binary(args.target_bin)

    arch_id = get_arch_id(arch)

    data_lines = []
    if args.target_func is not None and args.target_func in func_dict:
        data_lines.append({
            'hex': func_dict[args.target_func],
            'arch_id': arch_id
        })
        emb_list = gen_hex_embedding(device, data_lines, 1, tokenizer_name, config_name, model_name_or_path)
        print(f'[-]function:{args.target_func}, embedding:{emb_list[0]}')
        return

    for _, hex in func_dict.items():
        data_lines.append({
            'hex': hex,
            'arch_id': arch_id
        })
    emb_list = gen_hex_embedding(device, data_lines, BATCH_SIZE, tokenizer_name, config_name, model_name_or_path)
    
    func_list = list(func_dict.keys())
    func_emb_dict = {}
    for idx, func_name in enumerate(func_list):
        func_emb_dict[func_name] = emb_list[idx]

    with open(args.save_path, 'wb') as f:
        pickle.dump(func_emb_dict, f)

    print(f'[-] binary\'s function embeddings are saved in {args.save_path}')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='Asteria-Pro')
    ap.add_argument("--tokenizer_name", type=str, help="path to tokenizer")
    ap.add_argument("--config_name", type=str, help="path to config")
    ap.add_argument("--model_path", type=str, help="path to model")
    ap.add_argument("--target_func", type=str, help="target function name")
    ap.add_argument("--target_bin", type=str, help="path to target binary")
    ap.add_argument("--is_gpu", action="store_true", help="gpu usage")
    ap.add_argument("--save_path", type=str, help="save functions embedding as pkl")
    args = ap.parse_args()
    main(args)
