#!/usr/bin/env bash

set -uo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_NAME="${DATA_NAME:-train_annotated}"
DOC_START="${DOC_START:-0}"
DOC_END="${DOC_END:--1}"
HOST_ADDRESS="${HOST_ADDRESS:-127.0.0.1}"
PORT="${PORT:-6006}"
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
Usage: ./run_train_annotated_cache_pipeline.sh [options]

Options:
  --python-exe <path>      Python executable (default: python)
  --project-root <path>    Project root
  --data-name <name>       train_annotated or train (default: train_annotated)
  --doc-start <int>        DOC_START (default: 0)
  --doc-end <int>          DOC_END, -1 means full dataset (default: -1)
  --host-address <addr>    Model service host (default: 127.0.0.1)
  --port <int>             Model service port (default: 6006)
  --resume                 Resume from checkpoint
  --reset-checkpoint       Remove existing checkpoint first
  --checkpoint-file <path> Custom checkpoint file
  -h, --help               Show help
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
    printf "%-34s %-10s %-8s %s\n" "Step" "Status" "Seconds" "Log"
    local i
    for i in "${!STEP_NAMES[@]}"; do
        printf "%-34s %-10s %-8s %s\n" "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}" "${STEP_SECONDS[$i]}" "${STEP_LOGS[$i]}"
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
        --resume) RESUME=1; shift ;;
        --reset-checkpoint) RESET_CHECKPOINT=1; shift ;;
        --checkpoint-file) CHECKPOINT_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

if [[ "$DATA_NAME" != "train_annotated" && "$DATA_NAME" != "train" ]]; then
    fail "This script is for training-side cache generation only. Use --data-name train_annotated or train."
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
unset USE_RERANK || true
unset METHOD_TAG || true

LOGS_ROOT="${PROJECT_ROOT}/logs"
ensure_directory "$LOGS_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOGS_ROOT}/pipeline_${DATA_NAME}_cache_${RANGE_TAG}_${TIMESTAMP}"
ensure_directory "$RUN_LOG_DIR"

if [[ -z "${CHECKPOINT_FILE// }" ]]; then
    CHECKPOINT_FILE="${LOGS_ROOT}/pipeline_${DATA_NAME}_cache_${RANGE_TAG}.checkpoint.json"
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

REQUIRED_DIRS=(
    "${PROJECT_ROOT}/data/entity_information_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/entity_information_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/entity_information/${DATA_NAME}"
    "${PROJECT_ROOT}/data/relation_summary_prompt/${DATA_NAME}"
    "${PROJECT_ROOT}/data/relation_summary_run/${DATA_NAME}"
    "${PROJECT_ROOT}/data/check_result_relation_summary_jsonl/${DATA_NAME}"
    "${PROJECT_ROOT}/data/get_embeddings"
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
echo "Logs directory: $RUN_LOG_DIR"
echo "Checkpoint file: $CHECKPOINT_FILE"
if [[ "$RESUME" -eq 1 && "${#COMPLETED_STEPS[@]}" -gt 0 ]]; then
    echo "Resume mode: skipping ${#COMPLETED_STEPS[@]} completed step(s)"
fi

set -e
trap 'echo "Logs directory: $RUN_LOG_DIR"; print_summary' EXIT

invoke_step "01_entity_information_prompt" "${PROJECT_ROOT}/2.entity_information" "entity_information_prompt_new.py" "$RUN_LOG_DIR"
invoke_step "02_entity_information_run" "${PROJECT_ROOT}/2.entity_information" "entity_information_run.py" "$RUN_LOG_DIR" 1
invoke_step "03_entity_information_check" "${PROJECT_ROOT}/2.entity_information" "check_result_entity_information_jsonl.py" "$RUN_LOG_DIR"

invoke_step "04_relation_summary_prompt" "${PROJECT_ROOT}/3.relation_summary" "relation_summary_prompt.py" "$RUN_LOG_DIR"
invoke_step "05_relation_summary_run" "${PROJECT_ROOT}/3.relation_summary" "relation_summary_run.py" "$RUN_LOG_DIR" 1
invoke_step "06_relation_summary_check" "${PROJECT_ROOT}/3.relation_summary" "check_result_relation_summary_jsonl.py" "$RUN_LOG_DIR"

invoke_step "07_get_embeddings" "${PROJECT_ROOT}/4.retrieval" "get_embeddings.py" "$RUN_LOG_DIR"

echo ""
echo "Training-side cache pipeline completed successfully."
