import util
import json
import sys
import traceback
import re

def replace_start_words(text):
    # 使用正则表达式匹配以"好的"或"嗯"开头的字符串，并替换
    pattern = r'^(好的|嗯)'
    replaced_text = re.sub(pattern, '为什么这么回答，我是这么想的', text)
    return replaced_text


def pro(dic):
    dic['sys_prompt'] = dic['sys_prompt'] + '\n/no_think'
    #print (dic['res'])
    ans = dic['res'].split('</think>')[1].strip()
    think = dic['res'].split('</think>')[0].replace('<think>', '').strip()
    think = replace_start_words(think)
    #print (think)
    #print ('================================================')
    #dic['res'] = f'<think>\n\n</think>\n{ans}\n<my_think>\n{think}\n</my_think>'
    dic['res'] = f'<think>\n\n</think>\n{ans}\n<my_think>\n{think}所以，最终要输出的答案是：\n{ans}\n</my_think>'
    #dic['res'] = f'<think>\n\n</think>\n{ans}\n'
    return dic


if __name__ == '__main__':
    for line in sys.stdin:
        dic = json.loads(line.strip())
        try:
            out = pro(dic)
            print (json.dumps(out))
        except:
            sys.stderr.write(util.dic2json(dic))
            traceback.print_exc()