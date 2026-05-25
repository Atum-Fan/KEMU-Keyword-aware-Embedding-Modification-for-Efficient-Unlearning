#!/bin/bash
#SBATCH --gpus=8

source /etc/profile.d/modules.sh
module load miniforge3
module load cuda/12.4
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate opunlearning
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "$TRITON_CACHE_DIR" ]; then
    export TRITON_CACHE_DIR="/tmp/.triton_cache"
fi
mkdir -p $TRITON_CACHE_DIR

models=(
    "Llama-2-7b-chat-hf"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=1
gradient_accumulation_steps=16

# KEMU optimal parameters
OPTIMAL_TOPK=10
OPTIMAL_STEP_SIZE=0.07

get_master_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

run_kemu() {
    local forget_split=$1
    local holdout_split=$2
    local retain_split=$3
    local model=$4

    local task_name="tofu_${model}_${forget_split}_KEMU"
    local model_path="saves/finetune/tofu_${model}_full"
    local output_dir="saves/unlearn/${task_name}"

    echo ""
    echo "============================================================"
    echo "Starting Task: ${task_name}"
    echo "Model: ${model_path}"
    echo "Forget Split: ${forget_split}"
    echo "fisher_topk: ${OPTIMAL_TOPK}"
    echo "step_size: ${OPTIMAL_STEP_SIZE}"
    echo "============================================================"
    echo ""

    export MASTER_PORT=$(get_master_port)

    accelerate launch \
        --config_file configs/accelerate/kemu_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train_kemu.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=KEMU \
        task_name=${task_name} \
        model=${model} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
        trainer.args.ddp_find_unused_parameters=false \
        trainer.args.gradient_checkpointing=true \
        trainer.method_args.fisher_topk=${OPTIMAL_TOPK} \
        trainer.method_args.step_size=${OPTIMAL_STEP_SIZE} \
        trainer.method_args.norm_preserve=true \
        trainer.method_args.keyword_selection_method=frequency

    echo ""
    echo "Training completed for ${task_name}"
    echo ""

    echo "Starting evaluation for ${task_name}..."

    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${output_dir} \
        paths.output_dir=saves/unlearn/${task_name}/evals \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

    echo ""
    echo "Evaluation completed for ${task_name}"
    echo "============================================================"
    echo ""
}

main() {
    echo "============================================================"
    echo "TOFU KEMU Optimal Configuration Test"
    echo "============================================================"
    echo ""
    echo "Optimal Parameters:"
    echo "  fisher_topk: ${OPTIMAL_TOPK}"
    echo "  step_size: ${OPTIMAL_STEP_SIZE}"
    echo "============================================================"

    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)

        for model in "${models[@]}"; do
            run_kemu ${forget_split} ${holdout_split} ${retain_split} ${model}
        done
    done

    echo ""
    echo "============================================================"
    echo "TOFU KEMU optimal configuration test completed!"
    echo "============================================================"
}

main "$@"