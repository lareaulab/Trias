#!/bin/bash

WANDB_PROJECT=Trias python3 ../src/trias/main.py \
    --report_to wandb \
    --fp16 True \
    --gradient_checkpointing False \
    --seed 42 \
    --run_name trias \
    --max_seq_len 2048 \
    --dataset_name ../examples/dummy_dataset/ \
    --vocab_file ... \
    --output_dir ... \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8  \
    --learning_rate 1e-4 \
    --train_len 9956303 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 4 \
    --max_steps 650000 \
    --logging_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 3 \
