import sys
import json
import vllm_clien
from vllm_clien import call_llm
import util

if __name__ == '__main__':
    client = vllm_clien.VllmClient(sys_prompt='')
    file_name = sys.argv[1]
    outs = [['text', 'std_ans', 'result', 'my_think']]
    for line in sys.stdin:
        dic = json.loads(line.strip())
        client.session.clear()
        sys_prompt = dic['sys_prompt'] + '/no_think'
        #sys_prompt = dic['sys_prompt']
        text = dic['text']
        res = dic['res']
        print ('===============会话记录[START]===================')
        print (text)
        print ('===============会话记录[END]===================')
        reasoning, result = call_llm(client=client, prompt=text, sys_prompt=sys_prompt, 
            #stop=['<my_think>']
        )
        print ('===============参考答案[START]===================')
        print (res.split('</think>')[1])
        print ('===============参考答案[END]===================')
        tmp = [dic['text'], res.split('</think>')[1], result.split('<my_think>')[0], result.split('<my_think>')[1]]
        outs.append(tmp)
    #print (outs)
    util.list_to_csv(outs, file_name)
        