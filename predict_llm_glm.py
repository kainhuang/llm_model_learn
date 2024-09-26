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
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    #low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# 将模型设置为评估模式
model.eval()

session = []

def generate_response(prompt, max_tokens=256000):
    global session
    st = time.time()
    gen_kwargs = {"do_sample": True, "top_k": 1, "max_tokens": max_tokens}
    # 生成响应
    with torch.no_grad():
        pre_len = 0
        print ('Resp: ')
        for resp, history in model.stream_chat(tokenizer, prompt, session, role = 'user', **gen_kwargs):
            sys.stdout.write(resp[pre_len:])
            sys.stdout.flush()
            pre_len = len(resp)
    print ()
    #print ('History: %s' % history)
    session = history
    ed = time.time()
    print("模型响应时间: 总时间:%ss 生成效率%s token/s" % (ed-st, pre_len/(ed-st)))


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

