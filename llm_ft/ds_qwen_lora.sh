num_gpus=6
# 启动训练前设置环境变量
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_NCCL_TRACE_BUFFER_SIZE=10485760  # 10MB 缓冲区
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
# 使用更高性能的通信协议
export NCCL_PROTO=Simple  # 替代默认的 LL/LL128

# 调整网络超时参数
export NCCL_TIMEOUT=1800000  # 超时延长至 30 分钟 (单位：毫秒)

# 禁用 IB（InfiniBand）流控（若有 RDMA）
# export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
deepspeed --num_gpus $num_gpus llm_model_train_sft.py \
  --deepspeed ./ds_zero2.json \
  --pretrain_model_path  /root/autodl-tmp/modelscope/models/Qwen/Qwen3-8B/ \
  --data_path data/train6_1k.jsonl \
  --max_len 4096 \
  --use_lora True \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
  --lora_bias none \
  --output_dir output/Qwen3_8B_qa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --num_train_epochs 2 \
  --save_steps 1000 \
  --learning_rate 1e-5 \
  --save_on_each_node False \
  --gradient_checkpointing False \
  --save_only_model \
  --fp16

