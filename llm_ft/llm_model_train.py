from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict, Any
from torch.utils.data import Dataset
import json

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    pretrain_model_path: str = field(default="./pretrain_model/Qwen2-7B-Instruct/")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    max_len: int = 8192

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_bias: str = "none"


def preprocess(
    messages,
    tokenizer,
    max_len,
):
    """Preprocesses the data for supervised fine-tuning."""

    texts = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            #padding="max_length",
            max_length=max_len,
            truncation=True,
    )
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
    # 将JSON文件转换为CSV文件
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


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses() 
    # 打印解析后的参数
    print(model_args)
    print(data_args)
    print(lora_args)
    print(training_args)
    pretrain_model_path = model_args.pretrain_model_path
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, trust_remote_code=True)
    print ('Loading dataset %s' % data_args.data_path)
    tokenized_ds = make_dataset2(data_args.data_path, tokenizer, data_args.max_len)
    print ('Loading pretrain model %s' % pretrain_model_path)
    model = AutoModelForCausalLM.from_pretrained(pretrain_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    
    if training_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=lora_args.lora_target_modules,
            inference_mode=False, # 训练模式
            r=lora_args.lora_r, # Lora 秩
            lora_alpha=lora_args.lora_alpha, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=lora_args.lora_dropout,# Dropout 比例
            bias=lora_args.lora_bias
        )
        model = get_peft_model(model, lora_config)
    print ('Training...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()


if __name__ == '__main__':
    main()
