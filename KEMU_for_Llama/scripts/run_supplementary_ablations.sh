#!/bin/bash
#SBATCH --gpus=4

source /etc/profile.d/modules.sh
module load miniforge3
module load cuda/12.4
source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate opunlearning

set -euo pipefail

MODEL="Llama-2-7b-chat-hf"
RESULT_DIR="${PWD}/grid_search"
REF_MODEL_DIR="saves/finetune/tofu_${MODEL}_full"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
PARAM_EPSILON="1e-6"
PARAM_TOPK=20

# 1. Core optimal configs (frequency)
# Format: forget_split holdout_split retain_split topk step epochs keyword_method
best_kemu_configs=(
    "forget01 holdout01 retain99 12 0.100 3 frequency"
    "forget05 holdout05 retain95 43 0.0300 4 frequency"
    "forget10 holdout10 retain90 51 0.0320 4 frequency"
)

# 2. Keyword Selection Ablation
# Testing different keyword_method on optimal forget01 configs
keyword_ablation_configs=(
    "forget01 holdout01 retain99 12 0.100 3 random"
    "forget01 holdout01 retain99 12 0.100 3 pure_freq"
)

mkdir -p "${RESULT_DIR}"

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
    local keyword_method=$7

    # kw_xxx suffix prevents overwriting
    local task_name="tofu_${MODEL}_${forget_split}_KEMU_tk${topk}_st${step}_e${epochs}_kw_${keyword_method}"
    local result_json="${RESULT_DIR}/${task_name}.json"

    if should_skip_task "${result_json}"; then
        echo "[skip] ${task_name}"
        return
    fi

    echo "============================================================"
    echo "KEMU: ${task_name}"
    echo "  fisher_topk=${topk} | step_size=${step} | epochs=${epochs} | method=${keyword_method}"
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
        ++trainer.method_args.keyword_selection_method="${keyword_method}" \
        trainer.method_args.norm_preserve=true \
        trainer.method_args.pseudo_sample=false \
        trainer.args.num_train_epochs="${epochs}" \
        trainer.args.warmup_epochs=0 \
        trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        trainer.args.gradient_checkpointing=true \
        model.model_args.pretrained_model_name_or_path="${REF_MODEL_DIR}"

    run_eval_and_collect "${task_name}" "${forget_split}" "${holdout_split}" "${retain_split}"
    cleanup_checkpoint "${task_name}"
}

# ============================================================
# Execution
# ============================================================

echo ""
echo "============================================================"
echo "  Phase 1: Run current best KEMU configs (Frequency)"
echo "============================================================"
for cfg in "${best_kemu_configs[@]}"; do

    read -r forget_split holdout_split retain_split topk step epochs keyword_method <<< "$cfg"
    run_kemu "${forget_split}" "${holdout_split}" "${retain_split}" "${topk}" "${step}" "${epochs}" "${keyword_method}"
done

echo ""
echo "============================================================"
echo "  Phase 2: Run Keyword Selection Ablation"
echo "============================================================"
for cfg in "${keyword_ablation_configs[@]}"; do
    read -r forget_split holdout_split retain_split topk step epochs keyword_method <<< "$cfg"
    run_kemu "${forget_split}" "${holdout_split}" "${retain_split}" "${topk}" "${step}" "${epochs}" "${keyword_method}"
done

echo ""
echo "============================================================"
echo "  Aggregate results"
echo "============================================================"
python src/aggregate_results.py
