CUDA_VISIBLE_DEVICES=0 python -u llm_model_train_sft.py \
    --pretrain_model_path  /root/autodl-tmp/modelscope/models/Qwen/Qwen3-0.6B/ \
    --data_path data/train6_1w.jsonl \
    --max_len 8192 \
    --use_lora False \
    --output_dir output/Qwen3_0.6B_qa_lora2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --num_train_epochs 2 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --save_on_each_node True \
    --gradient_checkpointing True 

