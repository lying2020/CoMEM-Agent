#!/bin/bash

# Please set the model name according to your needs
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

export PYTHONPATH=GUI-Agent-Learn-From-Error/CoMEM-Agent-train:$PYTHONPATH

python src_agent/merge_lora_weights.py \
    --model-path '' \
    --model-base $MODEL_NAME  \
    --save-model-path '' \
    --safe-serialization