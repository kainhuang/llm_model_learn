from datasets import Dataset
import pandas as pd
import torch
from typing import Optional, Tuple, Union, List, Dict, Any
import json
import sys
from transformers.trainer_pt_utils import LabelSmoother
from util import dic2json, load_json, write_list_to_jsonl
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    messages,
    tokenizer,
    max_len,
):
    """Preprocesses the data for supervised fine-tuning."""
    """
    print (dic2json(messages))
   
    # 1. 先获取原始文本（不tokenize）
    raw_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 关键修改：不进行tokenize
        add_generation_prompt=False,
    )
    print("原始文本:\n", raw_text)  # 打印原始对话文本
    """
    texts = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            #padding="max_length",
            max_length=max_len,
            truncation=True,
    )
    if len(texts) > 8192:
        print ('long text len =', len(texts))
        #print("Token IDs:", texts)  # 可选：打印token IDs
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(
        input_ids=input_ids, labels=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_file, tokenizer, max_len
    ):
        super(SupervisedDataset, self).__init__()

        self.data = []
        for line in open(data_file):
            line = line.strip()
            messages = json.loads(line)
            tokenized_ids = preprocess(messages, tokenizer, max_len)
            self.data.append(tokenized_ids)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data[i]


def make_dataset2(data_file, tokenizer, max_len=512):
    ds = SupervisedDataset(data_file, tokenizer, max_len)
    return ds


def make_dataset(data_file, tokenizer, max_len=512):
    df = pd.read_json(data_file, lines=True)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input']},
            {"role": "assistant", "content": example['output']}
        ]
        ret = preprocess(messages, tokenizer, max_len)
        return ret

    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    return tokenized_ds


def make_dataset2(data_file):
    def process_func(example):
        messages = [
            {"role": "system", "content": example['sys_prompt']},
            {"role": "user", "content": example['text']},
            {"role": "assistant", "content": example['res']}
        ]
        ret = {"messages": messages}
        return ret

    ret_lis = []
    for line in open(data_file):
        ep = json.loads(line.strip())
        sp = process_func(ep)
        ret_lis.append(sp)
    return ret_lis


def make_dataset3(data_file, tokenizer, max_len=512):
    # 将JSON文件转换为CSV文件
    df = pd.read_json(data_file, lines=True)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        messages = [
            {"role": "system", "content": example['sys_prompt']},
            {"role": "user", "content": example['text']},
            {"role": "assistant", "content": example['res']}
        ]
        ret = preprocess(messages, tokenizer, max_len)
        return ret

    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    return tokenized_ds

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    datas = make_dataset2(input_file)
    # print (datas)
    write_list_to_jsonl(datas, output_file)
