#!/usr/bin/env bash

set -uo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_NAME="${DATA_NAME:-dev}"
DOC_START="${DOC_START:-0}"
DOC_END="${DOC_END:-100}"
HOST_ADDRESS="${HOST_ADDRESS:-127.0.0.1}"
PORT="${PORT:-6006}"
USE_RERANK_RAW="${USE_RERANK:-true}"
ENTITY_PAIR_SAMPLE_COUNT="${ENTITY_PAIR_SAMPLE_COUNT:-1}"
METHOD_TAG="${METHOD_TAG:-}"
RESUME=0
RESET_CHECKPOINT=0
CHECKPOINT_FILE=""

STEP_NAMES=()
STEP_STATUS=()
STEP_SECONDS=()
STEP_LOGS=()
COMPLETED_STEPS=()
SCRIPT_CHECKPOINT_FILE=""
RUN_LOG_DIR=""

usage() {
    cat <<'USAGE'
Usage: ./run_dev_pipeline.sh [options]

Options:
  --python-exe <path>                 Python executable (default: python)
  --project-root <path>               Project root
  --data-name <name>                  DATA_NAME (default: dev)
  --doc-start <int>                   DOC_START (default: 0)
  --doc-end <int>                     DOC_END, -1 means full dataset (default: 100)
  --host-address <addr>               Model service host (default: 127.0.0.1)
  --port <int>                        Model service port (default: 6006)
  --use-rerank <bool>                 true/false (default: true)
  --entity-pair-sample-count <int>    Sampling count for stage1 (default: 1, range 1-20)
  --method-tag <text>                 METHOD_TAG override
  --resume                            Resume from checkpoint
  --reset-checkpoint                  Remove existing checkpoint first
  --checkpoint-file <path>            Custom checkpoint file
  -h, --help                          Show help
USAGE
}

fail() {
    echo "$1" >&2
    exit 1
}

assert_path_exists() {
    local path_value="$1"
    local description="$2"
    [[ -e "$path_value" ]] || fail "$description not found: $path_value"
}

ensure_directory() {
    mkdir -p "$1"
}

convert_to_bool() {
    local value="${1:-}"
    local normalized
    normalized="$(echo "$value" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "$normalized" in
        1|true|yes|y|on) echo "true" ;;
        0|false|no|n|off) echo "false" ;;
        *) return 1 ;;
    esac
}

test_port_open() {
    local host_name="$1"
    local port_number="$2"
    "$PYTHON_EXE" - "$host_name" "$port_number" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.5)
try:
    sock.connect((host, port))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    sock.close()
PY
}

assert_model_service_available() {
    if ! test_port_open "$HOST_ADDRESS" "$PORT"; then
        fail "Model service is not reachable on ${HOST_ADDRESS}:${PORT}. Start it first."
    fi
}

assert_at_least_one_match() {
    local pattern="$1"
    local description="$2"
    if ! compgen -G "$pattern" > /dev/null; then
        fail "$description not found. Expected pattern: $pattern"
    fi
}

normalize_step_list_via_python() {
    "$PYTHON_EXE" - "$@" <<'PY'
import json
import os
import re
import sys

def normalize(raw_steps):
    result = []
    for raw in raw_steps:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        parts = [p.strip() for p in re.split(r'(?=\d{2}_)', text) if p and p.strip()]
        if not parts:
            parts = [text]
        for part in parts:
            if part not in result:
                result.append(part)
    return result

checkpoint_path = sys.argv[1]
candidates = [checkpoint_path, checkpoint_path + ".bak"]
existing = [p for p in candidates if os.path.exists(p)]
if not existing:
    sys.exit(0)

for candidate in existing:
    try:
        raw = open(candidate, "r", encoding="utf-8").read().strip()
        if not raw:
            continue
        data = json.loads(raw)
        steps = normalize(data.get("completed_steps", []))
        for step in steps:
            print(step)
        sys.exit(0)
    except Exception:
        continue

print(f"Failed to load checkpoint file: {checkpoint_path}", file=sys.stderr)
sys.exit(1)
PY
}

load_checkpoint() {
    local checkpoint_path="$1"
    local raw_steps
    if ! raw_steps="$(normalize_step_list_via_python "$checkpoint_path")"; then
        fail "Failed to load checkpoint file: $checkpoint_path"
    fi
    if [[ -z "$raw_steps" ]]; then
        COMPLETED_STEPS=()
    else
        mapfile -t COMPLETED_STEPS <<< "$raw_steps"
    fi
}

save_checkpoint() {
    local checkpoint_path="$1"
    local latest_log_dir="$2"
    shift 2
    "$PYTHON_EXE" - "$checkpoint_path" "$DATA_NAME" "$DOC_START" "$DOC_END" "$latest_log_dir" "$@" <<'PY'
import json
import os
import re
import shutil
import sys
from datetime import datetime

checkpoint_path = sys.argv[1]
data_name = sys.argv[2]
doc_start = int(sys.argv[3])
doc_end = int(sys.argv[4])
latest_log_dir = sys.argv[5]
raw_steps = sys.argv[6:]

def normalize(raw_steps):
    result = []
    for raw in raw_steps:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        parts = [p.strip() for p in re.split(r'(?=\d{2}_)', text) if p and p.strip()]
        if not parts:
            parts = [text]
        for part in parts:
            if part not in result:
                result.append(part)
    return result

payload = {
    "data_name": data_name,
    "doc_start": doc_start,
    "doc_end": doc_end,
    "latest_log_dir": latest_log_dir,
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "completed_steps": normalize(raw_steps),
}

tmp_path = checkpoint_path + ".tmp"
bak_path = checkpoint_path + ".bak"
dir_name = os.path.dirname(checkpoint_path) or "."
os.makedirs(dir_name, exist_ok=True)

with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

if os.path.exists(checkpoint_path):
    shutil.copy2(checkpoint_path, bak_path)

os.replace(tmp_path, checkpoint_path)
PY
}

append_step_result() {
    STEP_NAMES+=("$1")
    STEP_STATUS+=("$2")
    STEP_SECONDS+=("$3")
    STEP_LOGS+=("$4")
}

is_completed_step() {
    local step_name="$1"
    local item
    for item in "${COMPLETED_STEPS[@]}"; do
        [[ "$item" == "$step_name" ]] && return 0
    done
    return 1
}

safe_step_name() {
    echo "$1" | sed 's/[^A-Za-z0-9_-]/_/g'
}

invoke_step() {
    local name="$1"
    local working_directory="$2"
    local script_name="$3"
    local log_dir="$4"
    local requires_model_service="${5:-0}"
    local safe_name
    safe_name="$(safe_step_name "$name")"
    local log_file="${log_dir}/${safe_name}.log"
    local stdout_log="${log_dir}/${safe_name}.stdout.log"
    local stderr_log="${log_dir}/${safe_name}.stderr.log"
    local script_path="${working_directory}/${script_name}"
    local started_at duration_seconds exit_code

    assert_path_exists "$working_directory" "Working directory for $name"
    assert_path_exists "$script_path" "Script for $name"

    if is_completed_step "$name"; then
        echo ""
        echo "==== $name ===="
        echo "Skipping completed step from checkpoint."
        append_step_result "$name" "SKIPPED" "0" "$log_file"
        return 0
    fi

    if [[ "$requires_model_service" -eq 1 ]]; then
        assert_model_service_available
    fi

    echo ""
    echo "==== $name ===="
    echo "Working directory: $working_directory"
    echo "Command: $PYTHON_EXE $script_name"
    echo "Log: $log_file"

    started_at="$(date +%s)"
    rm -f "$stdout_log" "$stderr_log"

    set +e
    (
        cd "$working_directory" || exit 1
        "$PYTHON_EXE" "$script_path" > "$stdout_log" 2> "$stderr_log"
    )
    exit_code=$?
    set -e

    {
        [[ -f "$stdout_log" ]] && cat "$stdout_log"
        [[ -f "$stderr_log" ]] && cat "$stderr_log"
    } > "$log_file"

    [[ -s "$stdout_log" ]] && cat "$stdout_log"
    [[ -s "$stderr_log" ]] && cat "$stderr_log"

    duration_seconds="$(( $(date +%s) - started_at ))"
    if [[ "$exit_code" -ne 0 ]]; then
        echo "Step failed with exit code: $exit_code"
        append_step_result "$name" "FAILED" "$duration_seconds" "$log_file"
        return 1
    fi

    append_step_result "$name" "OK" "$duration_seconds" "$log_file"
    COMPLETED_STEPS+=("$name")
    save_checkpoint "$SCRIPT_CHECKPOINT_FILE" "$log_dir" "${COMPLETED_STEPS[@]}"
    echo "Step completed: $name (${duration_seconds}s)"
    return 0
}

print_summary() {
    if [[ "${#STEP_NAMES[@]}" -eq 0 ]]; then
        return
    fi
    echo ""
    echo "Step summary:"
    printf "%-42s %-10s %-8s %s\n" "Step" "Status" "Seconds" "Log"
    local i
    for i in "${!STEP_NAMES[@]}"; do
        printf "%-42s %-10s %-8s %s\n" "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}" "${STEP_SECONDS[$i]}" "${STEP_LOGS[$i]}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-exe) PYTHON_EXE="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --data-name) DATA_NAME="$2"; shift 2 ;;
        --doc-start) DOC_START="$2"; shift 2 ;;
        --doc-end) DOC_END="$2"; shift 2 ;;
        --host-address) HOST_ADDRESS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --use-rerank) USE_RERANK_RAW="$2"; shift 2 ;;
        --entity-pair-sample-count) ENTITY_PAIR_SAMPLE_COUNT="$2"; shift 2 ;;
        --method-tag) METHOD_TAG="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --reset-checkpoint) RESET_CHECKPOINT=1; shift ;;
        --checkpoint-file) CHECKPOINT_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

USE_RERANK="$(convert_to_bool "$USE_RERANK_RAW")" || fail "Invalid --use-rerank value: $USE_RERANK_RAW"

if ! [[ "$ENTITY_PAIR_SAMPLE_COUNT" =~ ^[0-9]+$ ]]; then
    fail "EntityPairSampleCount must be an integer."
fi
if (( ENTITY_PAIR_SAMPLE_COUNT < 1 || ENTITY_PAIR_SAMPLE_COUNT > 20 )); then
    fail "EntityPairSampleCount must be between 1 and 20."
fi

assert_path_exists "$PROJECT_ROOT" "Project root"
command -v "$PYTHON_EXE" > /dev/null 2>&1 || fail "Python executable not found: $PYTHON_EXE"

PYTHON_VERSION_OUTPUT="$("$PYTHON_EXE" --version 2>&1)" || fail "Unable to execute Python with: $PYTHON_EXE"

export DATA_NAME
export DOC_START="$DOC_START"
if (( DOC_END >= 0 )); then
    export DOC_END="$DOC_END"
    RANGE_TAG="${DOC_START}-${DOC_END}"
else
    unset DOC_END
    RANGE_TAG="${DOC_START}-full"
fi

export USE_RERANK
if [[ -z "${METHOD_TAG// }" ]]; then
    if [[ "$USE_RERANK" == "true" ]]; then
        export METHOD_TAG="rerank"
    else
        export METHOD_TAG="baseline"
    fi
else
    export METHOD_TAG="$METHOD_TAG"
fi

LOGS_ROOT="${PROJECT_ROOT}/logs"
ensure_directory "$LOGS_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOGS_ROOT}/pipeline_${DATA_NAME}_${RANGE_TAG}_${TIMESTAMP}"
ensure_directory "$RUN_LOG_DIR"

if [[ -z "${CHECKPOINT_FILE// }" ]]; then
    CHECKPOINT_FILE="${LOGS_ROOT}/pipeline_${DATA_NAME}_${RANGE_TAG}.checkpoint.json"
fi
SCRIPT_CHECKPOINT_FILE="$CHECKPOINT_FILE"

if [[ "$RESET_CHECKPOINT" -eq 1 && -f "$CHECKPOINT_FILE" ]]; then
    rm -f "$CHECKPOINT_FILE"
fi

if [[ "$RESUME" -eq 1 ]]; then
    load_checkpoint "$CHECKPOINT_FILE"
else
    COMPLETED_STEPS=()
    save_checkpoint "$SCRIPT_CHECKPOINT_FILE" "$RUN_LOG_DIR" "${COMPLETED_STEPS[@]}"
fi

assert_model_service_available
assert_at_least_one_match "${PROJECT_ROOT}/data/check_result_relation_summary_jsonl/train_annotated/result_docred_train_annotated_relation_summary_*.jsonl" "Train-side relation summary cache"
assert_at_least_one_match "${PROJECT_ROOT}/data/get_embeddings/docred_train_annotated_embeddings_*.npy" "Train-side embedding cache"

REQUIRED_DIRS=(
    "${PROJECT_ROOT}/data/entity_pair_selection_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/entity_pair_selection_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/check_result_entity_pair_selection_jsonl/${DATA_NAME}"
    "${PROJECT_ROOT}/data/get_entity_pair_selection_label/${DATA_NAME}"
    "${PROJECT_ROOT}/data/entity_information_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/entity_information_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/entity_information/${DATA_NAME}"
    "${PROJECT_ROOT}/data/relation_summary_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/relation_summary_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/check_result_relation_summary_jsonl/${DATA_NAME}"
    "${PROJECT_ROOT}/data/get_embeddings"
    "${PROJECT_ROOT}/data/retrieval_from_train/${DATA_NAME}"
    "${PROJECT_ROOT}/data/retrieval_rerank/${DATA_NAME}"
    "${PROJECT_ROOT}/data/multiple_choice_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/multiple_choice_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/check_result_multiple_choice_jsonl/${DATA_NAME}"
    "${PROJECT_ROOT}/data/get_multiple_choice_label/${DATA_NAME}"
    "${PROJECT_ROOT}/data/triplet_fact_judgement_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/triplet_fact_judgement_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/check_result_triplet_fact_judgement_jsonl/${DATA_NAME}"
    "${PROJECT_ROOT}/data/get_triplet_fact_judgement_label/${DATA_NAME}"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    ensure_directory "$dir"
done

echo "Python: $PYTHON_EXE"
echo "Python version: $PYTHON_VERSION_OUTPUT"
echo "Project root: $PROJECT_ROOT"
if (( DOC_END >= 0 )); then
    echo "Data range: $DATA_NAME $DOC_START-$DOC_END"
else
    echo "Data range: $DATA_NAME from $DOC_START to full dataset"
fi
echo "USE_RERANK: $USE_RERANK"
echo "METHOD_TAG: $METHOD_TAG"
echo "ENTITY_PAIR_SAMPLE_COUNT: $ENTITY_PAIR_SAMPLE_COUNT"
echo "Logs directory: $RUN_LOG_DIR"
echo "Checkpoint file: $CHECKPOINT_FILE"
if [[ "$RESUME" -eq 1 && "${#COMPLETED_STEPS[@]}" -gt 0 ]]; then
    echo "Resume mode: skipping ${#COMPLETED_STEPS[@]} completed step(s)"
fi

set -e
trap 'echo "Logs directory: $RUN_LOG_DIR"; print_summary' EXIT

invoke_step "01_entity_pair_selection_prompt" "${PROJECT_ROOT}/1.entity_pair_selection" "entity_pair_selection_prompt.py" "$RUN_LOG_DIR"
for (( sample_id=1; sample_id<=ENTITY_PAIR_SAMPLE_COUNT; sample_id++ )); do
    sample_tag="$(printf "%02d" "$sample_id")"
    export SAMPLE_TAG="$sample_tag"
    invoke_step "02_entity_pair_selection_run_${sample_tag}" "${PROJECT_ROOT}/1.entity_pair_selection" "entity_pair_selection_run.py" "$RUN_LOG_DIR" 1
    invoke_step "03_entity_pair_selection_check_${sample_tag}" "${PROJECT_ROOT}/1.entity_pair_selection" "check_result_entity_pair_selection_jsonl.py" "$RUN_LOG_DIR"
    invoke_step "04_entity_pair_selection_label_${sample_tag}" "${PROJECT_ROOT}/1.entity_pair_selection" "get_entity_pair_selection_label.py" "$RUN_LOG_DIR"
done
unset SAMPLE_TAG

invoke_step "05_entity_information_prompt" "${PROJECT_ROOT}/2.entity_information" "entity_information_prompt_new.py" "$RUN_LOG_DIR"
invoke_step "06_entity_information_run" "${PROJECT_ROOT}/2.entity_information" "entity_information_run.py" "$RUN_LOG_DIR" 1
invoke_step "07_entity_information_check" "${PROJECT_ROOT}/2.entity_information" "check_result_entity_information_jsonl.py" "$RUN_LOG_DIR"

invoke_step "08_relation_summary_prompt" "${PROJECT_ROOT}/3.relation_summary" "relation_summary_prompt.py" "$RUN_LOG_DIR"
invoke_step "09_relation_summary_run" "${PROJECT_ROOT}/3.relation_summary" "relation_summary_run.py" "$RUN_LOG_DIR" 1
invoke_step "10_relation_summary_check" "${PROJECT_ROOT}/3.relation_summary" "check_result_relation_summary_jsonl.py" "$RUN_LOG_DIR"

invoke_step "11_get_embeddings" "${PROJECT_ROOT}/4.retrieval" "get_embeddings.py" "$RUN_LOG_DIR"
invoke_step "12_retrieval_from_train" "${PROJECT_ROOT}/4.retrieval" "retrieval_from_train-few.py" "$RUN_LOG_DIR"
if [[ "$USE_RERANK" == "true" ]]; then
    invoke_step "13_evidence_relation_rerank" "${PROJECT_ROOT}/4.retrieval" "evidence_relation_rerank.py" "$RUN_LOG_DIR"
fi

invoke_step "14_multiple_choice_prompt" "${PROJECT_ROOT}/5.multiple_choice" "multiple_choice_prompt.py" "$RUN_LOG_DIR"
invoke_step "15_multiple_choice_run" "${PROJECT_ROOT}/5.multiple_choice" "multiple_choice_run.py" "$RUN_LOG_DIR" 1
invoke_step "16_multiple_choice_check" "${PROJECT_ROOT}/5.multiple_choice" "check_result_multiple_choice_jsonl.py" "$RUN_LOG_DIR"
invoke_step "17_multiple_choice_label" "${PROJECT_ROOT}/5.multiple_choice" "get_multiple_choice_label.py" "$RUN_LOG_DIR"

invoke_step "18_triplet_fact_judgement_prompt" "${PROJECT_ROOT}/6.triplet_fact_judgement" "triplet_fact_judgement_prompt.py" "$RUN_LOG_DIR"
invoke_step "19_triplet_fact_judgement_run" "${PROJECT_ROOT}/6.triplet_fact_judgement" "triplet_fact_judgement_run.py" "$RUN_LOG_DIR" 1
invoke_step "20_triplet_fact_judgement_check" "${PROJECT_ROOT}/6.triplet_fact_judgement" "check_result_triplet_fact_judgement_jsonl.py" "$RUN_LOG_DIR"
invoke_step "21_triplet_fact_judgement_label" "${PROJECT_ROOT}/6.triplet_fact_judgement" "get_triplet_fact_judgement_label.py" "$RUN_LOG_DIR"

echo ""
echo "Pipeline completed successfully."
