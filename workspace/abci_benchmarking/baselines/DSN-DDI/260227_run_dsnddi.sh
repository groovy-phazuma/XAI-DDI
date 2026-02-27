#!/bin/bash
#PBS -P gah51684
#PBS -q rt_HG
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -o fold3_job.log

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

# 実行
python /home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/baselines/DSN-DDI/260227_dsnddi_abci.py \
  --batch_size 128 \
  --n_epochs 200 \
  --fold_dir /home/aah18044co/github/XAI-DDI/dataset/inductive_data/fold3 \
  --out_dir /home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/baselines/DSN-DDI/results/260227/fold3
