#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# local/xjsonfile/rftV2
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
# export NCCL_SOCKET_IFNAME=eth1

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/yelp_grefer
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=29501 main.py  \
   --data_path  local/jsonfile-- \
   --data_split 10,0,0 \
   --model_name_or_path your_llama_path \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 2048 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --only_optimize_lora \
   --lora_dim 8 \
   --lora_module_name "layers." \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log 
