#!/bin/bash
#SBATCH --job-name=agml-confirm
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# Confirm the harness on items it was not tuned on: another seed picks 2 different items per dataset. Each repeat runs
# the no-tools baseline (pinned agbench, unchanged), look first, and look first with the local knowledge base and
# keep-healthy, one after another on one GPU, so any drift hits all three.
# usage: sbatch farm_confirm.sh <model> [seed, default 1] [repeats, default 3] [arms, default "baseline look-first kb-keep-healthy"] [out suffix]
# e.g. a later version of one arm next to earlier runs: sbatch farm_confirm.sh google/gemma-4-12B-it 1 3 kb-keep-healthy -searchfix
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$HOME/.local/share/uv/tools/vllm/bin:$PATH
module load cuda/13.3.0 && export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LLAMA_SERVER_BIN=/group/jmearlesgrp/scratch/eranario/llama.cpp/build/bin/llama-server
ls -d ~/.local/share/uv/tools/vllm/lib/python3.12/site-packages/flashinfer_cubin* >/dev/null 2>&1 || vllm download-kernels
nvidia-smi --query-gpu=name,memory.total --format=csv
MODEL=$1; SEED=${2:-1}; REPEATS=${3:-3}; ARMS=${4:-"baseline look-first kb-keep-healthy"}; SUFFIX=${5:-}
OUT=runs/confirm/seed$SEED
COMMON=(--model "$MODEL" --data-dir $DATA --judge --limit-per-dataset 2 --seed "$SEED")
HARNESS=(--retrieve-mode crosscheck-look-first --no-live-search --retrieve-cache runs/pilot_api/retrieve_cache.jsonl)
for i in $(seq "$REPEATS"); do
    for arm in $ARMS; do
        echo "=== repeat $i: $arm $(date)"
        case $arm in
            baseline) uv run --extra bench agbench eval run "${COMMON[@]}" --out $OUT/baseline$SUFFIX ;;
            look-first) uv run --extra bench agml-agent-bench "${COMMON[@]}" "${HARNESS[@]}" --out $OUT/look-first$SUFFIX ;;
            kb-keep-healthy) uv run --extra bench agml-agent-bench "${COMMON[@]}" "${HARNESS[@]}" --clean-text 2 --check-options \
                --kb kb --keep-healthy --out $OUT/kb-keep-healthy$SUFFIX ;;
            kb-tidy) uv run --extra bench agml-agent-bench "${COMMON[@]}" "${HARNESS[@]}" --clean-text 2 --check-options \
                --kb kb --keep-healthy --tidy-names --crop-from-prompt --out $OUT/kb-tidy$SUFFIX ;;
            kb-only) uv run --extra bench agml-agent-bench "${COMMON[@]}" --retrieve-mode crosscheck-look-first --clean-text 2 \
                --check-options --kb kb --keep-healthy --kb-only --out $OUT/kb-only$SUFFIX ;;  # local database only, no network
            *) echo "unknown arm $arm"; exit 1 ;;
        esac
    done
done
