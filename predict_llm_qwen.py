import time
import sys

if len(sys.argv) < 2:
    print('Usage: %s <model>' % sys.argv[0])
    sys.exit(0)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型和tokenizer的路径
model_path = sys.argv[1]


# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
# config = AutoConfig.from_pretrained(model_path)
# config.quantization_config["use_exllama"] = False
device_map = 'balanced_low_0'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # config=config,
    torch_dtype='auto',
    device_map=device_map,
    #low_cpu_mem_usage=True,
    trust_remote_code=True
)
# 将模型设置为评估模式
model.eval()
print (model.hf_device_map)
session = []

def generate_response(prompt, max_new_tokens=4096):
    st = time.time()
    session.append({"role": "user", "content": prompt})
    # 编码输入文本
    inputs = tokenizer.apply_chat_template(session,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                       attn_implementation="flash_attention_2"
                                       )
    print ('inputs.shape: ', inputs['input_ids'].shape)
    # print ('inputs: ', tokenizer.decode(inputs['input_ids'][0]))
    inputs.to(device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "top_k": 1}
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print ('outputs.shape: ', outputs.shape)
    # print ('response: ', tokenizer.decode(outputs[0]))
    # 解码生成的响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    session.append({"role": "assistant", "content": response})
    ed = time.time()
    print("模型响应时间: 总时间:%ss 生成效率%s token/s" % (ed-st, outputs.shape[1]/(ed-st)))
    return response


while True:
    # 示例输入
    input_list = []
    print ('input:...')
    while True:
        tmp = input()
        if tmp == 'eof':
            break
        if tmp == 'clear':
            session.clear()
            continue
        input_list.append(tmp)
    prompt = '\n'.join(input_list)
    # 生成响应
    print ('答案生成中...')
    response = generate_response(prompt)
    print("模型响应:", response)
