#!/bin/bash
#SBATCH --job-name=agml-baseline
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# No-tools baseline with the agbench pinned in this checkout (unchanged), on the same node type and setup as harness runs.
# usage: sbatch farm_run_baseline.sh <model> [extra agbench eval run args]     full run into runs/baseline-full
#        resume: sbatch farm_run_baseline.sh <model> --run-dir runs/baseline-full/<run dir>
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$HOME/.local/share/uv/tools/vllm/bin:$PATH
module load cuda/13.3.0 && export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LLAMA_SERVER_BIN=/group/jmearlesgrp/scratch/eranario/llama.cpp/build/bin/llama-server
ls -d ~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/flashinfer_cubin* >/dev/null 2>&1 || vllm download-kernels
nvidia-smi --query-gpu=name,memory.total --format=csv
MODEL=$1; shift
uv run --extra bench agbench eval run --model "$MODEL" --data-dir $DATA --judge --out runs/baseline-full "$@"
