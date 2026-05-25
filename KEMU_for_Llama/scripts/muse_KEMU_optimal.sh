#!/bin/bash
#SBATCH --gpus=8
#SBATCH --time=4:00:00

# ============================================================================
# MUSE KEMU - Optimal Configuration Test
# ============================================================================

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE any Python/CUDA initialization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source /etc/profile.d/modules.sh
module load miniforge3
module load cuda/12.4
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate opunlearning

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

# ============================================================================
# Configuration
# ============================================================================

if [ -z "$MODEL_PATH" ]; then
    export MODEL_PATH="${MODEL_PATH:-/data/home/user/run/pretrained_models/Llama-2-7b-chat-hf"
fi

echo "Using model from: $MODEL_PATH"

# KEMU pretrained model - using MUSE finetuned model
KEMU_PRETRAINED_MODEL="${MUSE_FINETUNED_MODEL_PATH:-${PROJECT_ROOT:-/data/home/user/run/openunlearning/openunlearning/open-unlearning-main}/saves/finetune/muse_Llama-2-7b}"

MODEL="Llama-2-7b-hf"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT:-/data/home/user/run/openunlearning/openunlearning/open-unlearning-main}/muse_optimal_results}"
mkdir -p ${RESULT_DIR}

DATA_SPLITS=("News")

# Training hyperparameters
per_device_train_batch_size=2
gradient_accumulation_steps=8

# ============================================================================
# Optimal KEMU Parameters
# ============================================================================

# Best config: tk16_st30_e1_lr1e-5
OPTIMAL_TOPK=16
OPTIMAL_STEERING=30
OPTIMAL_EPOCHS=1
OPTIMAL_LR="1e-5"

# ============================================================================
# Helper Functions
# ============================================================================

get_master_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

# ============================================================================
# KEMU Training Function
# ============================================================================

run_kemu_optimal() {
    local data_split=$1
    local topk=$2
    local steering=$3
    local epochs=$4
    local lr=$5

    local task_name="muse_${MODEL}_${data_split}_KEMU_OPTIMAL_tk${topk}_st${steering}_e${epochs}_lr${lr}"
    local output_dir="saves/unlearn/${task_name}"

    if [ -f "${RESULT_DIR}/${task_name}.json" ]; then
        echo "[skip] ${task_name} (already exists)"
        return
    fi

    echo "============================================================"
    echo "KEMU Optimal Configuration Test"
    echo "============================================================"
    echo "Task: ${task_name}"
    echo "  fisher_topk=${topk}"
    echo "  step_size=${steering}"
    echo "  epochs=${epochs}"
    echo "  learning_rate=${lr}"
    echo "============================================================"

    export MASTER_PORT=$(get_master_port)

    accelerate launch \
        --config_file configs/accelerate/kemu_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train_kemu.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model.model_args.pretrained_model_name_or_path=${KEMU_PRETRAINED_MODEL} \
        trainer=KEMU \
        task_name=${task_name} \
        data_split=${data_split} \
        trainer.method_args.fisher_topk=${topk} \
        trainer.method_args.step_size=${steering} \
        trainer.args.learning_rate=${lr} \
        trainer.args.num_train_epochs=${epochs} \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.ddp_find_unused_parameters=false

    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=Llama-2-7b-hf-local \
        model.model_args.pretrained_model_name_or_path=${output_dir} \
        model.tokenizer_args.pretrained_model_name_or_path=${output_dir} \
        paths.output_dir=saves/unlearn/${task_name}/evals

    # Compute parameter changes
    echo "[metrics] Computing parameter changes..."
    python src/param_change.py ${task_name} \
        --reference_model ${KEMU_PRETRAINED_MODEL} \
        --target_model ${output_dir} \
        --output saves/unlearn/${task_name}/evals/PARAM_CHANGE.json \
        --epsilon 1e-6 \
        --topk 20

    python src/results.py ${task_name}

    # Copy result to result directory
    cp "grid_search/${task_name}.json" "${RESULT_DIR}/" 2>/dev/null || true

    # Cleanup
    echo "[cleanup] ${task_name}"
    rm -f ${output_dir}/*.safetensors
    rm -f ${output_dir}/*.bin

    echo "[done] ${task_name}"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo "============================================================"
    echo "MUSE KEMU - Optimal Configuration Test"
    echo "============================================================"
    echo ""
    echo "Optimal Parameters (from grid search analysis):"
    echo "  fisher_topk: ${OPTIMAL_TOPK}"
    echo "  step_size: ${OPTIMAL_STEERING}"
    echo "  epochs: ${OPTIMAL_EPOCHS}"
    echo "  learning_rate: ${OPTIMAL_LR}"
    echo ""
    echo "Expected Performance:"
    echo "  EM: ~0.114"
    echo "  FK: ~0.009"
    echo "  RK: ~0.527"
    echo "  Trade-off Score: ~0.413"
    echo "============================================================"

    for data_split in "${DATA_SPLITS[@]}"; do
        run_kemu_optimal ${data_split} ${OPTIMAL_TOPK} ${OPTIMAL_STEERING} ${OPTIMAL_EPOCHS} ${OPTIMAL_LR}
    done

    echo ""
    echo "============================================================"
    echo "KEMU optimal configuration test completed!"
    echo "Results saved to: ${RESULT_DIR}"
    echo "============================================================"
}

# Run main
main "$@"
