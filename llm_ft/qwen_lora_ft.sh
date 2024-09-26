python -u llm_model_train.py \
    --pretrain_model_path  pretrain_model/Qwen2-7B-Instruct/ \
    --data_path dataset/HuatuoGPT-sft-data-v1/huatuo_ft_example_3w.jsonl \
    --max_len 4096 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
    --lora_bias none \
    --output_dir output/Qwen2_7B_huatuo_lora \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --num_train_epochs 3 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --save_on_each_node True \
    --gradient_checkpointing True 

