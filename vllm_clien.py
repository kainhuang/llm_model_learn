from openai import OpenAI
import sys
from typing import Optional, Tuple, Union, List, Dict, Any
import time


class VllmClient(object):
    def __init__(self, sys_prompt = '', openai_api_key = "EMPTY", openai_api_base = "http://localhost:8000/v1/"):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = self.client.models.list()
        self.model = models.data[0].id
        self.sys_prompt = sys_prompt
        self.session = []

    def generate(self, prompt: str,
                sys_prompt=None,
                stream=True,
                max_tokens: int = 1024,
                stop: List[str] = [], #生成到这些字符串的时候停止
                n=1,  #生成的补全数量，这里是 1
                presence_penalty=0.0,
                frequency_penalty=0.0,
                top_p=0.8,
                temperature=0.1
                ):
        st = time.time()
        if sys_prompt is None:
            sys_prompt = self.sys_prompt
        if self.session is None or len(self.session) == 0:
            self.session = [{"role": "system", "content": sys_prompt}]

        self.session.append({"role": "user", "content": prompt})
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.session,
            stream=stream,
            n=n,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            temperature=temperature,
            stop=stop,
        )
        if stream:
            for chat in chat_response:
                #print (chat)
                if hasattr(chat.choices[0].delta, 'reasoning_content'):
                    reasoning_content = chat.choices[0].delta.reasoning_content
                    if reasoning_content:
                        yield reasoning_content, 'reasoning'

                content = chat.choices[0].delta.content
                if content:
                    yield content, 'content'
        else:
            for chat in chat_response:
                print (chat)
            return None


SYS_PROMPT = """你是一个万能的AI助手，你叫Ein，由kain开发
"""

def call_llm(client, prompt, sys_prompt = None, stop = []):
    client.session.clear()
    first_word = True
    st = time.time()
    reasoning = []
    res = []
    for word, tp in client.generate(prompt=prompt, sys_prompt=sys_prompt, max_tokens=8192, stop=stop):
        if first_word:
            first_word_cost = time.time() - st       
            first_word = False
        #print (word, tp)
        if tp == 'reasoning':
            reasoning.append(word)
        if tp == 'content':
            res.append(word)
        sys.stdout.write(word)
        sys.stdout.flush()
    print()
    res = ''.join(res)
    reasoning = ''.join(reasoning)
    client.session.append({"role": "assistant", "content": res})
    #print (client.session)
    cost = time.time() - st
    print ('tot_cost_time[%s], first_word_cost[%s] char_per_s[%s] res[%s]' % (cost, first_word_cost, len(res) / cost, len(res)))
    return reasoning, res


def run1():
    client = VllmClient(sys_prompt = '')
    while True:
        # 示例输入
        input_list = []
        print ('input:...')
        while True:
            tmp = input()
            if tmp == 'eof':
                break
            if tmp == 'clear':
                client.session.clear()
                continue
            input_list.append(tmp)
        prompt = '\n'.join(input_list)
        call_llm(client=client, prompt=prompt)


def run2():
    client = VllmClient(sys_prompt = '')
    while True:
        # 示例输入
        input_list = []
        print ('input:...')
        while True:
            tmp = input()
            if tmp == 'eof':
                break
            if tmp == 'clear':
                client.session.clear()
                continue
            input_list.append(tmp)
        prompt = '\n'.join(input_list)
        first_word = True
        st = time.time()
        res = client.generate(prompt=prompt, stream=False)
        print (res)

    
if __name__ == '__main__':
    run1()