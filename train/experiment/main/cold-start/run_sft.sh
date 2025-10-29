export DEBUG_MODE="true"
export LOG_PATH="/train/outputs/Main/ColdStart/debug_log.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    train/experiment/main/cold-start/train/sft.py \
    --output_dir "train/outputs/Main/ColdStart/Qwen2.5-VL-7B-coldstart-sft" \
    --model_name_or_path "model/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "data/train_data/Llava-cot-data/cold_start_data_finally.json" \
    --deepspeed train/local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name Qwen2.5-VL-7B-coldstart-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model false \