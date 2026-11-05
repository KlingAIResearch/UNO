# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

GPUS=8

model_path='path_to_model'
output_path="path_to_output"

# generate images using the precomputed thinking reasoning data
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp.py \
    --output_dir $output_path/images \
    --metadata_file eval/gen/geneval2/geneval2_data_think.jsonl \
    --batch_size 1 \
    --num_images 4 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \


python eval/gen/geneval/evaluate_geneval2.py \
    --benchmark_data GenEval2/geneval2_data.jsonl \
    --method soft_tifa_am \
    --data_path $output_path \
    --num_images 4 \
    --output_file $output_path/score_lists.json
