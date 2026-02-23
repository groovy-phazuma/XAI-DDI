#!/bin/bash
#PBS -P gah51684
#PBS -q rt_HG
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o job.log

set -euo pipefail

cd "$PBS_O_WORKDIR"

echo "===== JOB ENV ====="
hostname
date

# (例) venv/conda など環境を有効化
module purge
module load python/3.12/3.12.9
source /home/aah18044co/github/XAI-DDI/benchmark_env_py312/bin/activate
python -V
which python

# wandb (おすすめ: offline から始める)
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_START_METHOD=thread
export WANDB_DIR="$PBS_O_WORKDIR/wandb_cache"
mkdir -p "$WANDB_DIR"

# 実行
python /home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/260223_train_route2_abci.py \
  --wandb_mode offline \
  --batch_size 128 \
  --n_epochs 200 \
  --out_dir /home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/results/260223/fold1
