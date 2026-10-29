#!/bin/bash
#SBATCH --job-name=agml-prefill
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
# Fill the retrieve cache with every classification class name (CPU only, network-bound); see bench_prefill.py.
# The default cache is the one full runs (--out runs/harness) read. Rerun to resume (low is preemptible).
# usage: sbatch farm_prefill.sh [extra agml-agent-bench-prefill args]
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$PATH
uv run --extra bench agml-agent-bench-prefill --data-dir $DATA --cache runs/harness/retrieve_cache.jsonl "$@"
