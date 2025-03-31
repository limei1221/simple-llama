#!/bin/bash
set -e

model_dir=/workspace/simple-llama/exp/Llama-3.2-0.3B_pretrain_fineweb_Rope_GQA

cp $model_dir/*.json $model_dir/checkpoint-10000/
python inference.py \
    --model_path $model_dir/checkpoint-10000 \
    --prompt "Once upon a time, there was a cat." \
    --max_length 256 \
    --temperature 0.8 \
    --top_k 40 \
    --top_p 0.95 \
    --num_return_sequences 3
