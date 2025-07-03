# 18GiB * 2
nproc_per_node=5

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /root/autodl-tmp/modelscope/models/Qwen/Qwen3-14B/ \
    --train_type full \
    --dataset ./data/train_msg_1w.jsonl\
    --val_dataset ./data/train_msg_1k.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 8192 \
    --output_dir output/Qwen3-14B_qa_sft \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers $nproc_per_node \
    --model_author swift \
    --model_name swift-robot \
    --swanlab_project swift-robot \
    --deepspeed zero3 \
    --save_only_model True