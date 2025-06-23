#!/bin/bash

# ============================ GPU Settings ===============================
# To run on specific GPUs, set the CUDA_VISIBLE_DEVICES environment variable.
# For example, to use GPUs 0, 1, 2, and 3:
export CUDA_VISIBLE_DEVICES=0,1,2,3
# The --devices argument below should match the number of GPUs you've made visible (in this case, 4).
# ========================================================================

dataset="mimic_cxr"
annotation="/nfs/scratch/eechengyang/Data/mimic-cxr/mimic_annotation_all.json"
base_dir="/nfs/scratch/eechengyang/Data/mimic-cxr/images"

version="v1_shallow"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 8 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --learning_rate 1e-4 \
    --gradient_clip_val 1 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 4 \
    --max_epochs 5 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt