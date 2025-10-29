#!/bin/bash
# run_models.sh


model_paths=(
    ""
)


######### Dataset ###########
datasets=(
    ""
)


methods=(
    "PCDA"
)
# Options:
# Think + Answer TA
# Prethink + Caption + Deepthink + Answer PCDA


savePaths=(
    ""
)


tips=(
    "dft"
)


export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    dataset="${datasets[$i]}"
    method="${methods[$i]}"
    savepath="${savePaths[$i]}"
    tip="${tips[$i]}"
    CUDA_VISIBLE_DEVICES=0 python /eval/eval_benchmark.py --model_path "$model" --dataset "$dataset" --cot "$method" --savepath "$savepath" --tips "$tip"
done
