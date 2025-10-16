#!/usr/bin/env bash
# run_ci_tests.sh — CI-like test runner for interactive environments.
# Reproduces the GitHub CI pipeline locally or inside Docker:
# - Lint: pre-commit 3.6.0
# - Unit tests: pytest with coverage
# - Functional tests: training (DDP; optional inprocess restart via ft_launcher), converter, models, recipes
# - Aggregates and reports coverage
#
# Modes:
# - local  (default): uses system Python environment
# - docker: builds docker/Dockerfile.ci and runs launch scripts in GPU container
#
# Requirements:
# - local: Python 3.10+ with pip; for functional tests, NVIDIA GPUs + CUDA, PyTorch distributed
# - docker: Docker with GPU runtime (nvidia-container-toolkit) available
#
# Environment variables:
# - HF_HOME: Hugging Face cache directory (default: <repo>/.hf_home)
# - CUDA_VISIBLE_DEVICES: GPU ids to use (default: 0,1)
# - GH_TOKEN: GitHub token used by tests/tools requiring GitHub API (required)
#
# Examples:
#   bash scripts/run_ci_tests.sh
#   bash scripts/run_ci_tests.sh --mode docker
#   bash scripts/run_ci_tests.sh --gpus 0 --skip-functional
#   bash scripts/run_ci_tests.sh --skip-lint --skip-unit
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

MODE="local"           # local | docker
SKIP_LINT="false"
SKIP_UNIT="false"
SKIP_FUNCTIONAL="false"
CUDA_DEVICES_DEFAULT="0,1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES_DEFAULT}}
HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_home"}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode [local|docker]     Run tests locally (python) or inside Docker (default: local)
  --skip-lint               Skip lint/pre-commit step
  --skip-unit               Skip unit tests
  --skip-functional         Skip functional tests
  --gpus <ids>              Set CUDA_VISIBLE_DEVICES (default: ${CUDA_DEVICES_DEFAULT})
  --hf-home <path>          Set HF_HOME cache directory (default: "+${REPO_ROOT}/.hf_home+")
  -h, --help                Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --mode docker
  $(basename "$0") --gpus 0 --skip-functional
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --skip-lint)
      SKIP_LINT="true"
      shift 1
      ;;
    --skip-unit)
      SKIP_UNIT="true"
      shift 1
      ;;
    --skip-functional)
      SKIP_FUNCTIONAL="true"
      shift 1
      ;;
    --gpus)
      CUDA_VISIBLE_DEVICES="${2:-}"
      shift 2
      ;;
    --hf-home)
      HF_HOME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

export HF_HOME
export CUDA_VISIBLE_DEVICES

# Require GH_TOKEN to be set for operations that need GitHub API access.
if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "[env] GH_TOKEN is not set. Please export GH_TOKEN before running this script." >&2
  exit 1
fi

run_lint_local() {
  if [[ "${SKIP_LINT}" == "true" ]]; then
    echo "[lint] Skipped"
    return 0
  fi
  echo "[lint] Installing and running pre-commit (3.6.0)"
  python -m pip install --upgrade pre-commit==3.6.0 "coverage[toml]"
  pre-commit install
  pre-commit run --all-files --show-diff-on-failure --color=always
}

run_unit_local() {
  if [[ "${SKIP_UNIT}" == "true" ]]; then
    echo "[unit] Skipped"
    return 0
  fi
  echo "[unit] Running unit tests with coverage"
  coverage erase || true
  coverage run -a -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
}

run_functional_local() {
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then
    echo "[functional] Skipped"
    return 0
  fi

  echo "[functional] Training group (excluding inprocess restart)"
  python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run -a -m pytest \
    -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/training -k "not test_inprocess_restart"

  if command -v ft_launcher >/dev/null 2>&1; then
    echo "[functional] Inprocess restart with ft_launcher"
    export TORCH_CPP_LOG_LEVEL="error"
    ft_launcher \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
      --nnodes=1 --nproc-per-node=2 \
      --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
      --ft-param-rank_out_of_section_timeout=300 \
      --monitor-interval=5 --max-restarts=3 \
      --ft-restart-policy=min-healthy \
      -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
      tests/functional_tests/training/test_inprocess_restart.py
  else
    echo "[functional] ft_launcher not found; skipping inprocess restart test"
  fi

  echo "[functional] Converter group"
  coverage run -a -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/converter

  echo "[functional] Models group"
  coverage run -a -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/models

  echo "[functional] Recipes group (2 GPUs)"
  python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run -a -m pytest \
    -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
    tests/functional_tests/recipes
}

run_local() {
  echo "[env] Using HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  run_lint_local
  run_unit_local
  run_functional_local
  echo "[coverage] Combine & report"
  coverage combine -q || true
  coverage report -i
}

run_docker() {
  if [[ "${SKIP_LINT}" == "true" ]]; then LINT_CMD="true"; else LINT_CMD="python -m pip install -U pre-commit==3.6.0 coverage[toml] && pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always"; fi
  if [[ "${SKIP_UNIT}" == "true" ]]; then UNIT_CMD="true"; else UNIT_CMD="bash tests/unit_tests/Launch_Unit_Tests.sh"; fi
  if [[ "${SKIP_FUNCTIONAL}" == "true" ]]; then FUNC_CMD="true"; else FUNC_CMD="bash tests/functional_tests/L2_Launch_training.sh && bash tests/functional_tests/L2_Launch_converter.sh && bash tests/functional_tests/L2_Launch_models.sh && bash tests/functional_tests/L2_Launch_recipes.sh"; fi

  echo "[docker] Building image from docker/Dockerfile.ci"
  docker build -f "${REPO_ROOT}/docker/Dockerfile.ci" -t megatron-bridge "${REPO_ROOT}"

  HOST_HF_HOME="${HF_HOME}"
  CONTAINER_HF_HOME="/home/TestData/HF_HOME"
  mkdir -p "${HOST_HF_HOME}"

  echo "[docker] Running tests in container (HF_HOME=${CONTAINER_HF_HOME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  docker run --rm -it --gpus all \
    -e HF_HOME="${CONTAINER_HF_HOME}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -e GH_TOKEN="${GH_TOKEN}" \
    -v "${REPO_ROOT}":/workspace \
    -v "${HOST_HF_HOME}":"${CONTAINER_HF_HOME}" \
    -w /workspace \
    megatron-bridge bash -lc "${LINT_CMD} && ${UNIT_CMD} && ${FUNC_CMD} && coverage report -i"
}

case "${MODE}" in
  local)
    run_local
    ;;
  docker)
    run_docker
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 2
    ;;
esac

echo "[done]"


