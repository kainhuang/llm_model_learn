import time
import sys
from llm_model import LLMModel

if len(sys.argv) < 2:
    print('Usage: %s <model>' % sys.argv[0])
    sys.exit(0)


# 指定模型和tokenizer的路径
model_path = sys.argv[1]

llm_model = LLMModel(model_path, '你是一个万能的智能助理，你叫Ein，由kain开发')


if __name__ == '__main__':
    session = None
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
        stop_token_ids = [151329, 151336, 151338]
        response, session = llm_model.generate_response(prompt, session, stop_token_ids=stop_token_ids)
        session.append({"role": "assistant", "content": response})
        print("模型响应:", response)

