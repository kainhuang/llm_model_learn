CUDA_VISIBLE_DEVICES=0 python -u llm_model_train.py \
    --pretrain_model_path  /root/autodl-tmp/modelscope/models/Qwen/Qwen3-14B/ \
    --data_path data/train6_1w.jsonl \
    --max_len 8192 \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
    --lora_bias none \
    --output_dir output/Qwen3_14B_qa_lora \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --save_on_each_node True \
    --gradient_checkpointing True 

