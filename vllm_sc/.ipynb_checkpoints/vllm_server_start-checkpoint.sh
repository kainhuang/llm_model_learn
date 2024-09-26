nohup python -m vllm.entrypoints.openai.api_server --model $1 --trust_remote_code --max_model_len 32000 > log &
