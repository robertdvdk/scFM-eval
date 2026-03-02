#!/usr/bin/env bash
# Create isolated venvs for foundation models with conflicting dependencies.
#
# Usage:
#   bash envs/setup_envs.sh            # Set up all model environments
#   bash envs/setup_envs.sh scgpt      # Set up only scGPT
#   bash envs/setup_envs.sh geneformer # Set up only Geneformer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=("scgpt" "geneformer" "scfoundation" "cancerfoundation")

# If a specific model is requested, only set up that one
if [[ $# -gt 0 ]]; then
    MODELS=("$1")
fi

for model in "${MODELS[@]}"; do
    model_dir="${SCRIPT_DIR}/${model}"
    venv_dir="${model_dir}/.venv"
    req_file="${model_dir}/requirements.txt"

    if [[ ! -f "$req_file" ]]; then
        echo "ERROR: requirements.txt not found at ${req_file}" >&2
        exit 1
    fi

    echo "=== Setting up ${model} environment at ${venv_dir} ==="

    # Per-model Python version override (e.g. scgpt needs 3.12 for torchtext)
    python_version_file="${model_dir}/python-version"
    uv_venv_args=("$venv_dir")
    if [[ -f "$python_version_file" ]]; then
        py_ver="$(cat "$python_version_file" | tr -d '[:space:]')"
        echo "  Using Python ${py_ver} (from ${python_version_file})"
        uv_venv_args+=(--python "$py_ver")
    fi

    # Create venv
    if [[ -d "$venv_dir" ]]; then
        echo "  Venv already exists, reinstalling dependencies..."
    else
        echo "  Creating venv..."
        uv venv "${uv_venv_args[@]}"
    fi

    # Install dependencies
    echo "  Installing dependencies from ${req_file}..."
    uv pip install --python "${venv_dir}/bin/python" -r "$req_file"

    echo "  Done: ${model}"
    echo ""
done

echo "All environments set up. Use with:"
echo "  python src/main.py task=batch_integration model=scgpt model.env_path=${SCRIPT_DIR}/scgpt/.venv"
