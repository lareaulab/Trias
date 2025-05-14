#!/bin/bash

WANDB_PROJECT=Trias python3 src/trias/main.py \
    --model_name_or_path lareaulab/Trias \
    --report_to wandb \
    --fp16 True \
    --gradient_checkpointing False \
    --seed 42 \
    --run_name trias \
    --max_seq_len 2048 \
    --dataset_name examples/dummy_dataset/ \
    --output_dir ouput/ \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8  \
    --learning_rate 1e-4 \
    --train_len 4 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --max_steps 10 \
    --logging_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 2 \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 3 \
