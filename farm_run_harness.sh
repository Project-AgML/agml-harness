#!/bin/bash
#SBATCH --job-name=agml-harness
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# Harness (model + tools) on AGgentic-benchmarks via the pinned agbench; mirrors AGgentic-benchmarks/farm_run.sh.
# Run from the agml-harness checkout (mkdir -p farm_outputs first; .env holds EPPO_API_KEY).
# usage: sbatch farm_run_harness.sh <model> [extra agml-agent-bench args]   full run with judge into runs/harness
#        pilot:  sbatch farm_run_harness.sh <model> --out runs/pilot --limit-per-dataset 2 --only '^(...)$'
#        resume: sbatch farm_run_harness.sh <model> --run-dir runs/harness/<run dir>
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$HOME/.local/share/uv/tools/vllm/bin:$PATH
module load cuda/13.3.0 && export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LLAMA_SERVER_BIN=/group/jmearlesgrp/scratch/eranario/llama.cpp/build/bin/llama-server
ls -d ~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/flashinfer_cubin* >/dev/null 2>&1 || vllm download-kernels
nvidia-smi --query-gpu=name,memory.total --format=csv
MODEL=$1; shift
# `--kb DIR`: read the knowledge base from a copy on the node's local disk (on the shared filesystem, lookups slowed
# from ~1 s to minutes once the index grew to 2.5 GB, 2026-11-09); falls back to DIR if the copy fails.
ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--kb" ] && [ -n "$2" ]; then
        LOCAL=${TMPDIR:-/tmp}/agml_kb_${SLURM_JOB_ID:-$$}
        if mkdir -p "$LOCAL" && cp "$2"/kb.sqlite "$2"/names.sqlite "$2"/manifest.json "$LOCAL"/ 2>/dev/null; then
            trap 'rm -rf "$LOCAL"' EXIT; echo "knowledge base copied to $LOCAL"; ARGS+=(--kb "$LOCAL")
        else
            rm -rf "$LOCAL"; echo "knowledge base: using $2 (local copy failed)"; ARGS+=(--kb "$2")
        fi
        shift 2
    else
        ARGS+=("$1"); shift
    fi
done
uv run --extra bench agml-agent-bench --model "$MODEL" --data-dir $DATA --judge --out runs/harness "${ARGS[@]}"
