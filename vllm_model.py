import time
import sys

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import Optional, Tuple, Union, List, Dict, Any

class LLMModel(object):
    def __init__(self, model_path: str,
                 sys_prompt:str = '', max_model_len = 32000):
        self.model_path = model_path
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        self.llm = LLM(model=model_path, trust_remote_code=True, max_model_len=max_model_len)
        self.sys_prompt = sys_prompt


    def generate_response(self, prompt: str,
                          session: List[str] = None, 
                          max_tokens: int = 32000,
                          stop_sequences: List[str] = None,
                          stop_token_ids: List[int] = None):
        st = time.time()
        if session is None or len(session) == 0:
            session = [{"role": "system", "content": self.sys_prompt}]
        session.append({"role": "user", "content": prompt})
        # 编码输入文本
        inputs = self.tokenizer.apply_chat_template(session,
                                    add_generation_prompt=True,
                                    tokenize=False,
                                    )
        print ('inputs', inputs)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_tokens, stop=stop_sequences, stop_token_ids=stop_token_ids)
        # 生成响应
        outputs = self.llm.generate([inputs], sampling_params)
        #print ('outputs: ', outputs)

        response = outputs[0].outputs[0].text
        token_ids = outputs[0].outputs[0].token_ids
        ed = time.time()
        print("模型响应时间: 总时间:%ss 生成效率%s token/s" % (ed-st, len(token_ids)/(ed-st)))
        return response, session


    def batch_generate_response(self, inputs: List[str], 
            max_tokens: int = 32000,
            stop_sequences: List[str] = None,
            stop_token_ids: List[int] = None
            ):
        st = time.time()
        sessions = []
        id_map = {}
        for i in range(len(inputs)):
            inp = inputs[i]
            id = inp.get('id', '0')
            id_map[i] = id
            prompt = inp.get('prompt', '')
            session = inp.get('session', [])
            if len(session) == 0:
                session.append({"role": "system", "content": self.sys_prompt})
            session.append({"role": "user", "content": prompt})
            sessions.append(session)

        # 编码输入文本
        inputs = self.tokenizer.apply_chat_template(sessions,
                                    add_generation_prompt=True,
                                    tokenize=False,
                                    )
        print ('inputs', inputs)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_tokens, stop=stop_sequences, stop_token_ids=stop_token_ids)
        # 生成响应
        outputs = self.llm.generate(inputs, sampling_params)
        #print ('outputs: ', outputs)
        tot_token_ids = 0
        ret_dic_list = []
        for i in range(len(outputs)):
            id = id_map[i]
            output = outputs[i]
            response = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            tot_token_ids += len(token_ids)
            sessions[i].append({"role": "assistant", "content": response})
            dic = {
                'id': id,
                'response': response,
                'session': sessions[i]
            }
            ret_dic_list.append(dic)
        ed = time.time()
        print("模型响应时间: 总时间:%ss 生成效率%s token/s" % (ed-st, tot_token_ids/(ed-st)))
        return ret_dic_list


if __name__ == '__main__':
    llm = LLMModel('pretrain_model/Qwen2-7B-Instruct/')
    inputs = [
            {'id':'1', 'prompt':'你好', 'session':[]},
            {'id':'2', 'prompt':'长江', 'session':[]},
            {'id':'2', 'prompt':'奥林匹克', 'session':[]}
            ]
    outputs = llm.batch_generate_response(inputs)
    print ('outputs: ',outputs)
    while True:
        inputs = []
        for output in outputs:
            prompt = input('input id[%s] prompt' % output['id'])
            output['prompt'] = prompt
            inputs.append(output)
        outputs = llm.batch_generate_response(inputs)
        print ('outputs: ',outputs)
 
