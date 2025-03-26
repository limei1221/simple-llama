#!/bin/bash
set -e

model_dir=/workspace/simple-llama/exp/Llama-3.2-0.3B_pretrain_fineweb/checkpoint-10000

cp /workspace/simple-llama/exp/Llama-3.2-0.3B_pretrain_fineweb/*.json $model_dir
python inference.py \
    --model_path $model_dir \
    --prompt "Once upon a time, there was a cat." \
    --max_length 256 \
    --temperature 0.8 \
    --top_k 40 \
    --top_p 0.95 \
    --num_return_sequences 3
