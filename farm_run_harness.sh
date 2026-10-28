#!/bin/bash
#SBATCH --job-name=agml-harness
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=jmearlesgrp
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
uv run --extra bench agml-agent-bench --model "$MODEL" --data-dir $DATA --judge --out runs/harness "$@"
