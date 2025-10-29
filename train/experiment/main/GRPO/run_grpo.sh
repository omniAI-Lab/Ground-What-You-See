cd /train/experiment/EXPERIMENTS/Main/GRPO

export DEBUG_MODE="true"
export LOG_PATH="/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO/debug_log.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    /train/experiment/EXPERIMENTS/Main/GRPO/grpo.py \
    --output_dir "/train/outputs/Main/GRPO/Qwen2.5-VL-7B-GRPO" \
    --model_name_or_path '/model/Qwen2.5-VL-7B-Instruct' \
    --dataset_name "/data/train_data/OmniVisionData/Train_RL_Medium_24K.json" \
    --deepspeed /train/local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name OmniVideoR1-GRPO-Training \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8 
