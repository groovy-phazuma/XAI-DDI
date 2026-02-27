#!/bin/bash
#PBS -P gah51684
#PBS -q rt_HG
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -o muffin_tvt_job.log
#PBS -J 1-5

set -euo pipefail
cd "$PBS_O_WORKDIR"

echo "===== JOB ENV ====="
hostname
date
nvidia-smi || true

# env activate（あなたの環境に合わせて）
module purge
module load python/3.12/3.12.9
source /home/aah18044co/github/XAI-DDI/benchmark_env_py312/bin/activate
python -V
which python

FOLD="${PBS_ARRAY_INDEX}"
echo "FOLD=${FOLD}"

python /home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/baselines/MUFFIN/260227_run_muffin.py \
  --mode train \
  --fold "${FOLD}" \
  --n_epoch 100 \
  --evaluate_every 10 \
  --val_ratio 0.1 \
  --DDI_batch_size 4096 \
  --DDI_evaluate_size 4096 \
  --lr 1e-3 \
  --select_metric auc \
  --out_root /home/aah18044co/github/XAI-DDI/workspace/abci_benchmarking/baselines/MUFFIN/results/260227/fold"${FOLD}"