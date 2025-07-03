# LoRA
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen3-0.6B \
    --adapters ./output/v2-20250512-111607/checkpoint-135 \
    --stream true \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048