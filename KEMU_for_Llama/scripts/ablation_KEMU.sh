#!/bin/bash
#SBATCH --gpus=4

source /etc/profile.d/modules.sh
module load miniforge3
module load cuda/12.4
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate opunlearning

set -euo pipefail

# ============================================================
# General Settings
# ============================================================
MODEL="Llama-2-7b-chat-hf"
RESULT_DIR="${PWD}/ablation_results"
REF_MODEL_DIR="saves/finetune/tofu_${MODEL}_full"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
PARAM_EPSILON="1e-6"
PARAM_TOPK=20

# Ablation study configuration
FORGET_SPLIT="forget01"
HOLDOUT_SPLIT="holdout01"
RETAIN_SPLIT="retain99"

mkdir -p "${RESULT_DIR}"

# ============================================================
# Helper Functions
# ============================================================
reserve_master_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

result_has_resource_usage() {
    local result_json=$1
    if [ ! -f "${result_json}" ]; then
        return 1
    fi

    python - "${result_json}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    resource = data.get("resource_usage", {})
    ok = (
        resource.get("modified_params_b") is not None
        and resource.get("peak_gpu_memory_gb") is not None
        and resource.get("gpu_hours") is not None
    )
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

should_skip_task() {
    local result_json=$1
    if result_has_resource_usage "${result_json}"; then
        return 0
    fi
    return 1
}

run_param_change_and_merge() {
    local task_name=$1
    local output_dir="saves/unlearn/${task_name}"
    local eval_dir="${output_dir}/evals"

    CUDA_VISIBLE_DEVICES=0 python src/param_change.py \
        "${task_name}" \
        --reference_model_dir "${REF_MODEL_DIR}" \
        --target_model_dir "${output_dir}" \
        --output "${eval_dir}/PARAM_CHANGE.json" \
        --epsilon "${PARAM_EPSILON}" \
        --topk "${PARAM_TOPK}"

    python src/results.py "${task_name}"
}

run_eval_and_collect() {
    local task_name=$1
    local forget_split=$2
    local holdout_split=$3
    local retain_split=$4

    local output_dir="saves/unlearn/${task_name}"
    local eval_dir="${output_dir}/evals"

    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split="${forget_split}" \
        holdout_split="${holdout_split}" \
        model="${MODEL}" \
        task_name="${task_name}" \
        model.model_args.pretrained_model_name_or_path="${output_dir}" \
        paths.output_dir="${eval_dir}" \
        retain_logs_path="saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json"

    run_param_change_and_merge "${task_name}"
}

cleanup_checkpoint() {
    local task_name=$1
    local output_dir="saves/unlearn/${task_name}"

    rm -f "${output_dir}"/*.safetensors
    rm -f "${output_dir}"/*.bin
}

run_kemu() {
    local forget_split=$1
    local holdout_split=$2
    local retain_split=$3
    local topk=$4
    local step=$5
    local epochs=$6
    local norm_preserve=$7
    local ablation_name=$8

    local task_name="tofu_${MODEL}_${forget_split}_KEMU_${ablation_name}"
    local result_json="${RESULT_DIR}/${task_name}.json"

    if should_skip_task "${result_json}"; then
        echo "[skip] ${task_name}"
        return
    fi

    echo "============================================================"
    echo "KEMU: ${task_name}"
    echo "  fisher_topk=${topk} | step_size=${step} | epochs=${epochs} | norm_preserve=${norm_preserve}"
    echo "============================================================"

    export MASTER_PORT
    MASTER_PORT=$(reserve_master_port)

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file configs/accelerate/kemu_config.yaml \
        --main_process_port "${MASTER_PORT}" \
        src/train_kemu.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=KEMU \
        task_name="${task_name}" \
        model="${MODEL}" \
        forget_split="${forget_split}" \
        retain_split="${retain_split}" \
        trainer.method_args.fisher_topk="${topk}" \
        trainer.method_args.step_size="${step}" \
        trainer.method_args.norm_preserve="${norm_preserve}" \
        trainer.method_args.pseudo_sample=false \
        trainer.args.num_train_epochs="${epochs}" \
        trainer.args.warmup_epochs=0 \
        model.model_args.pretrained_model_name_or_path="${REF_MODEL_DIR}" \
        trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        trainer.args.gradient_checkpointing=true

    run_eval_and_collect "${task_name}" "${forget_split}" "${holdout_split}" "${retain_split}"
    cleanup_checkpoint "${task_name}"
}

# ============================================================
# Ablation Study Configurations
# ============================================================

# Phase 1: Main Paper Ablations (7 experiments)
echo ""
echo "============================================================"
echo "  Phase 1: Main Paper Ablations (7 experiments)"
echo "============================================================"

# 1. Step Size Ablation (5 experiments)
# Arithmetic sequence: [0.02, 0.04, 0.06, 0.08, 0.10] with d=0.02
step_size_ablation=(
    "10 0.02 1 true ss0.02"
    "10 0.04 1 true ss0.04"
    "10 0.06 1 true ss0.06"
    "10 0.08 1 true ss0.08"
    "10 0.10 1 true ss0.10"
)

echo ""
echo "--- Step Size Ablation (Arithmetic Sequence, d=0.02) ---"
for cfg in "${step_size_ablation[@]}"; do
    topk=$(echo "${cfg}" | awk '{print $1}')
    step=$(echo "${cfg}" | awk '{print $2}')
    epochs=$(echo "${cfg}" | awk '{print $3}')
    norm_preserve=$(echo "${cfg}" | awk '{print $4}')
    ablation_name=$(echo "${cfg}" | awk '{print $5}')
    run_kemu "${FORGET_SPLIT}" "${HOLDOUT_SPLIT}" "${RETAIN_SPLIT}" "${topk}" "${step}" "${epochs}" "${norm_preserve}" "${ablation_name}"
done

# 2. Norm-Preserving Ablation (2 experiments)
norm_preserve_ablation=(
    "10 0.07 1 true np_true"
    "10 0.07 1 false np_false"
)

echo ""
echo "--- Norm-Preserving Ablation (Comparison) ---"
for cfg in "${norm_preserve_ablation[@]}"; do
    topk=$(echo "${cfg}" | awk '{print $1}')
    step=$(echo "${cfg}" | awk '{print $2}')
    epochs=$(echo "${cfg}" | awk '{print $3}')
    norm_preserve=$(echo "${cfg}" | awk '{print $4}')
    ablation_name=$(echo "${cfg}" | awk '{print $5}')
    run_kemu "${FORGET_SPLIT}" "${HOLDOUT_SPLIT}" "${RETAIN_SPLIT}" "${topk}" "${step}" "${epochs}" "${norm_preserve}" "${ablation_name}"
done

# Phase 2: Appendix Ablations (11 experiments)
echo ""
echo "============================================================"
echo "  Phase 2: Appendix Ablations (11 experiments)"
echo "============================================================"

# 3. Fisher Top-K Ablation (4 experiments)
# Approximate geometric sequence: [5, 15, 30, 60] with ratio ~2
fisher_topk_ablation=(
    "5 0.07 1 true fk5"
    "15 0.07 1 true fk15"
    "30 0.07 1 true fk30"
    "60 0.07 1 true fk60"
)

echo ""
echo "--- Fisher Top-K Ablation (Geometric Sequence, r~2) ---"
for cfg in "${fisher_topk_ablation[@]}"; do
    topk=$(echo "${cfg}" | awk '{print $1}')
    step=$(echo "${cfg}" | awk '{print $2}')
    epochs=$(echo "${cfg}" | awk '{print $3}')
    norm_preserve=$(echo "${cfg}" | awk '{print $4}')
    ablation_name=$(echo "${cfg}" | awk '{print $5}')
    run_kemu "${FORGET_SPLIT}" "${HOLDOUT_SPLIT}" "${RETAIN_SPLIT}" "${topk}" "${step}" "${epochs}" "${norm_preserve}" "${ablation_name}"
done

# 4. Epochs Ablation (5 experiments)
# Arithmetic sequence: [2, 4, 6, 8, 10] with d=2
epochs_ablation=(
    "10 0.07 2 true ep2"
    "10 0.07 4 true ep4"
    "10 0.07 6 true ep6"
    "10 0.07 8 true ep8"
    "10 0.07 10 true ep10"
)

echo ""
echo "--- Epochs Ablation (Arithmetic Sequence, d=2) ---"
for cfg in "${epochs_ablation[@]}"; do
    topk=$(echo "${cfg}" | awk '{print $1}')
    step=$(echo "${cfg}" | awk '{print $2}')
    epochs=$(echo "${cfg}" | awk '{print $3}')
    norm_preserve=$(echo "${cfg}" | awk '{print $4}')
    ablation_name=$(echo "${cfg}" | awk '{print $5}')
    run_kemu "${FORGET_SPLIT}" "${HOLDOUT_SPLIT}" "${RETAIN_SPLIT}" "${topk}" "${step}" "${epochs}" "${norm_preserve}" "${ablation_name}"
done

# ============================================================
# Aggregate Results
# ============================================================
echo ""
echo "============================================================"
echo "  Aggregate Results"
echo "============================================================"
python src/aggregate_results.py

echo ""
echo "============================================================"
echo "  KEMU Ablation Study Completed!"
echo "============================================================"
echo "Total experiments: 18"
echo "  - Main paper: 7 (Step Size + Norm-Preserving)"
echo "  - Appendix: 11 (Fisher Top-K + Epochs)"
echo ""
echo "Results saved to: ${RESULT_DIR}"
echo "============================================================"
