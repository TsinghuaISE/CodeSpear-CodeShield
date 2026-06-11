#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# User config
# ============================================================

# Inference script path
SCRIPT="Attack/LocalModels/script/run_main.py"

# Merge script path
MERGE_SCRIPT="Attack/LocalModels/script/merge_shards.py"

# Number of shards is auto-selected by model index; edit MODEL_SHARDS_MAP to change rules.

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# Supports space or comma separators
RUN_MODEL_INDEX="${RUN_MODEL_INDEX:-0 1 2}"
STRATEGY="${STRATEGY:-0 1}"
BENCHMARK="${BENCHMARK:-total}"
RUN_ROUNDS="${RUN_ROUNDS:-1}"

CODELANG="${CODELANG:-py}"

# Whether to suppress generation output
QUIET_GENERATION="${QUIET_GENERATION:-true}"

MERGE_AFTER="${MERGE_AFTER:-true}"

MERGE_CLEANUP="${MERGE_CLEANUP:-true}"

MERGE_PRUNE_RUN_INFO="${MERGE_PRUNE_RUN_INFO:-true}"

# When pruning run_info, keep latest (false = keep earliest)
MERGE_PREFER_LATEST_RUN_INFO="${MERGE_PREFER_LATEST_RUN_INFO:-false}"

# Log directory
LOG_DIR="${LOG_DIR:-logs_sharded}"
mkdir -p "${LOG_DIR}"

declare -A MODEL_DIR_MAP=(
  [0]="Meta-Llama-3-8B-Instruct"
  [1]="Qwen2.5-7B-Instruct"
  [2]="Qwen2.5-Coder-7B-Instruct"
)


declare -A MODEL_SHARDS_MAP=(
  [0]=8
  [1]=8
  [2]=8
)


# ============================================================
# Helper
# ============================================================

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  local arr=( ${raw} )
  printf "%s\n" "${arr[@]}"
}

expand_benchmarks() {
  local raw_list=("$@")
  local out=()

  for b in "${raw_list[@]}"; do
    if [[ "${b}" == "total" ]]; then
      out+=("rmc" "mal")
    else
      out+=("${b}")
    fi
  done

  # Deduplicate and preserve order
  local seen=""
  for b in "${out[@]}"; do
    if [[ " ${seen} " != *" ${b} "* ]]; then
      echo "${b}"
      seen="${seen} ${b}"
    fi
  done
}

bool_arg_enabled() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" ]]
}

get_num_shards_for_model() {
  local mid="$1"

  if [[ -z "${MODEL_SHARDS_MAP[$mid]:-}" ]]; then
    echo "[Error] MODEL_SHARDS_MAP does not contain model index: ${mid}" >&2
    echo "        Please add it to MODEL_SHARDS_MAP." >&2
    exit 1
  fi

  echo "${MODEL_SHARDS_MAP[$mid]}"
}

mapfile -t RUN_MODEL_INDEX_ARR < <(parse_list "${RUN_MODEL_INDEX}")
mapfile -t STRATEGY_ARR        < <(parse_list "${STRATEGY}")
mapfile -t BENCHMARK_RAW_ARR   < <(parse_list "${BENCHMARK}")
mapfile -t BENCHMARK_ARR       < <(expand_benchmarks "${BENCHMARK_RAW_ARR[@]}")
mapfile -t RUN_ROUNDS_ARR      < <(parse_list "${RUN_ROUNDS}")


# ============================================================
# Check config
# ============================================================

if [[ "${CODELANG}" != "py" && "${MERGE_AFTER}" == "true" ]]; then
  echo "[Error] merge_shards.py currently only matches _py_ result files."
  echo "        Please set CODELANG=py or modify merge_shards.py to support ${CODELANG}."
  exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[Error] SCRIPT not found: ${SCRIPT}"
  exit 1
fi

if [[ "${MERGE_AFTER}" == "true" && ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[Error] MERGE_SCRIPT not found: ${MERGE_SCRIPT}"
  exit 1
fi

IFS=',' read -ra GPUS <<< "${GPU_IDS}"
TOTAL_GPUS=${#GPUS[@]}

if (( TOTAL_GPUS != 8 )); then
  echo "[Error] This script expects exactly 8 GPUs, but got TOTAL_GPUS=${TOTAL_GPUS}: ${GPU_IDS}"
  exit 1
fi

for mid in "${RUN_MODEL_INDEX_ARR[@]}"; do
  if [[ -z "${MODEL_DIR_MAP[$mid]:-}" ]]; then
    echo "[Error] MODEL_DIR_MAP does not contain model index: ${mid}"
    echo "        Please add it to MODEL_DIR_MAP."
    exit 1
  fi

  model_num_shards="$(get_num_shards_for_model "${mid}")"

  if [[ "${model_num_shards}" != "8" && "${model_num_shards}" != "4" && "${model_num_shards}" != "2" && "${model_num_shards}" != "1" ]]; then
    echo "[Error] NUM_SHARDS for model ${mid} must be one of: 8, 4, 2, 1, got ${model_num_shards}"
    exit 1
  fi

  if (( TOTAL_GPUS % model_num_shards != 0 )); then
    echo "[Error] TOTAL_GPUS=${TOTAL_GPUS} cannot be evenly divided by NUM_SHARDS=${model_num_shards} for model ${mid}"
    exit 1
  fi
done

echo "============================================================"
echo "SCRIPT: ${SCRIPT}"
echo "MERGE_SCRIPT: ${MERGE_SCRIPT}"
echo "MODEL_SHARDS_MAP: 0/2/5/8/9/10/11/12=>8, 1=>1, 3/6=>4, 4/7=>2"
echo "GPU_IDS: ${GPU_IDS}"
echo "RUN_MODEL_INDEX: ${RUN_MODEL_INDEX_ARR[*]}"
echo "RUN_ROUNDS: ${RUN_ROUNDS_ARR[*]}"
echo "STRATEGY: ${STRATEGY_ARR[*]}"
echo "BENCHMARK(raw): ${BENCHMARK_RAW_ARR[*]}"
echo "BENCHMARK(expanded): ${BENCHMARK_ARR[*]}"
echo "CODELANG: ${CODELANG}"
echo "QUIET_GENERATION: ${QUIET_GENERATION}"
echo "MERGE_AFTER: ${MERGE_AFTER}"
echo "MERGE_CLEANUP: ${MERGE_CLEANUP}"
echo "MERGE_PRUNE_RUN_INFO: ${MERGE_PRUNE_RUN_INFO}"
echo "LOG_DIR: ${LOG_DIR}"
echo "============================================================"


# ============================================================
# Launch shards
# ============================================================

pids=()

cleanup() {
  echo
  echo "[Info] Caught interrupt. Killing child processes..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

safe_strategies="${STRATEGY// /-}"
safe_strategies="${safe_strategies//,/-}"
safe_benchmarks="${BENCHMARK// /-}"
safe_benchmarks="${safe_benchmarks//,/-}"

for mid in "${RUN_MODEL_INDEX_ARR[@]}"; do
  NUM_SHARDS="$(get_num_shards_for_model "${mid}")"
  GPUS_PER_SHARD=$(( TOTAL_GPUS / NUM_SHARDS ))

  echo
  echo "============================================================"
  echo "[Info] Launch model ${mid}: ${MODEL_DIR_MAP[$mid]}"
  echo "       NUM_SHARDS=${NUM_SHARDS}, GPUS_PER_SHARD=${GPUS_PER_SHARD}"
  echo "============================================================"

  pids=()

  for (( shard_id=0; shard_id<NUM_SHARDS; shard_id++ )); do
    start=$(( shard_id * GPUS_PER_SHARD ))
    end=$(( start + GPUS_PER_SHARD - 1 ))

    shard_gpus=""
    for (( i=start; i<=end; i++ )); do
      if [[ -z "${shard_gpus}" ]]; then
        shard_gpus="${GPUS[$i]}"
      else
        shard_gpus="${shard_gpus},${GPUS[$i]}"
      fi
    done

    log_file="${LOG_DIR}/shard${shard_id}of${NUM_SHARDS}_model-${mid}_strategies-${safe_strategies}_bench-${safe_benchmarks}.log"

    cmd=(
      python "${SCRIPT}"
      --run_model_index "${mid}"
      --run_rounds "${RUN_ROUNDS_ARR[@]}"
      --strategy "${STRATEGY_ARR[@]}"
      --benchmark "${BENCHMARK_RAW_ARR[@]}"
      --codelang "${CODELANG}"
      --num_shards "${NUM_SHARDS}"
      --shard_id "${shard_id}"
    )

    if bool_arg_enabled "${QUIET_GENERATION}"; then
      cmd+=(--quiet_generation)
    fi

    echo "[Launch] model ${mid} shard ${shard_id}/${NUM_SHARDS}"
    echo "         CUDA_VISIBLE_DEVICES=${shard_gpus}"
    echo "         log=${log_file}"
    echo "         cmd=${cmd[*]}"

    CUDA_VISIBLE_DEVICES="${shard_gpus}" "${cmd[@]}" > "${log_file}" 2>&1 &
    pids+=("$!")
  done

  echo
  echo "[Info] Model ${mid} all shards launched. Waiting..."
  echo "PIDs: ${pids[*]}"

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done

  if [[ "${fail}" -ne 0 ]]; then
    echo "[Error] Some shard processes failed for model ${mid}. Check logs in ${LOG_DIR}/"
    exit 1
  fi

  echo "[Done] Model ${mid} shards finished successfully."
done

echo "[Done] All models finished successfully."


# ============================================================
# Merge shards
# ============================================================

if bool_arg_enabled "${MERGE_AFTER}"; then
  echo
  echo "============================================================"
  echo "[Info] Start merging shard results..."
  echo "============================================================"

  merge_fail=0

  for mid in "${RUN_MODEL_INDEX_ARR[@]}"; do
    model_dir="${MODEL_DIR_MAP[$mid]}"

    for run_round in "${RUN_ROUNDS_ARR[@]}"; do
      for strategy in "${STRATEGY_ARR[@]}"; do
        for benchmark in "${BENCHMARK_ARR[@]}"; do

          safe_model_dir="${model_dir//\//_}"
          merge_log="${LOG_DIR}/merge_model-${mid}_${safe_model_dir}_s${strategy}_${benchmark}_round${run_round}.log"

          merge_cmd=(
            python "${MERGE_SCRIPT}"
            --strategy "${strategy}"
            --benchmark "${benchmark}"
            --model_dir "${model_dir}"
            --run_round "${run_round}"
          )

          if bool_arg_enabled "${MERGE_CLEANUP}"; then
            merge_cmd+=(--cleanup)
          fi

          if bool_arg_enabled "${MERGE_PRUNE_RUN_INFO}"; then
            merge_cmd+=(--prune-run-info)
          fi

          if bool_arg_enabled "${MERGE_PREFER_LATEST_RUN_INFO}"; then
            merge_cmd+=(--prefer-latest-run-info)
          fi

          echo "[Merge] model=${model_dir} | strategy=${strategy} | benchmark=${benchmark} | round=${run_round}"
          echo "        log=${merge_log}"
          echo "        cmd=${merge_cmd[*]}"

          if ! "${merge_cmd[@]}" > "${merge_log}" 2>&1; then
            echo "[Error] Merge failed: model=${model_dir}, strategy=${strategy}, benchmark=${benchmark}, round=${run_round}"
            echo "        Check log: ${merge_log}"
            merge_fail=1
          fi

        done
      done
    done
  done

  if [[ "${merge_fail}" -ne 0 ]]; then
    echo "[Error] Some merge jobs failed. Check logs in ${LOG_DIR}/"
    exit 1
  fi

  echo "[Done] All shard results merged successfully."
fi