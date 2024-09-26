import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

 
 
def merge_lora(model_path, lora_path, output_path):
    print(f"Loading the base model from {model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
 
    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    model_path = sys.argv[1]
    lora_path = sys.argv[2]
    output_path = sys.argv[3]
    merge_lora(model_path, lora_path, output_path)
