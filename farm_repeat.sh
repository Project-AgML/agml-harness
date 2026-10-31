#!/bin/bash
#SBATCH --job-name=agml-repeat
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=jmearlesgrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# Repeat the no-tools baseline and the crosscheck harness on the pilot items, to measure run-to-run noise (temperature
# 1.0). Runs alternate on one GPU, so any drift hits both. The baseline uses the agbench pinned here, unchanged.
# usage: sbatch farm_repeat.sh <model> [repeats, default 3]
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$HOME/.local/share/uv/tools/vllm/bin:$PATH
module load cuda/13.3.0 && export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LLAMA_SERVER_BIN=/group/jmearlesgrp/scratch/eranario/llama.cpp/build/bin/llama-server
ls -d ~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/flashinfer_cubin* >/dev/null 2>&1 || vllm download-kernels
nvidia-smi --query-gpu=name,memory.total --format=csv
MODEL=$1; REPEATS=${2:-3}
for i in $(seq "$REPEATS"); do
    echo "=== repeat $i: baseline $(date)"
    uv run --extra bench agbench eval run --model "$MODEL" --data-dir $DATA --judge --limit-per-dataset 2 --out runs/repeat/baseline
    echo "=== repeat $i: crosscheck $(date)"
    uv run --extra bench agml-agent-bench --model "$MODEL" --data-dir $DATA --judge --limit-per-dataset 2 --out runs/repeat/crosscheck \
        --retrieve-mode crosscheck --no-live-search --retrieve-cache runs/pilot_api/retrieve_cache.jsonl
done
