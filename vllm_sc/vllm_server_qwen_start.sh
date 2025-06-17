#vllm serve /root/autodl-tmp/modelscope/models/Qwen/QwQ-32B \
vllm serve /root/autodl-tmp/llm_model_learn/llm_ft/output/Qwen3_8B_qa_lora_merge
--port 8000 \
--max_model_len 32000
