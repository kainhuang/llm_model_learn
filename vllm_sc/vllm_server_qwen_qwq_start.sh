vllm serve /root/autodl-tmp/modelscope/models/Qwen/QwQ-32B \
--port 8000 \
--reasoning-parser deepseek_r1 \
--max_model_len 32000 \
--enable-auto-tool-choice \
--tool-call-parser hermes
