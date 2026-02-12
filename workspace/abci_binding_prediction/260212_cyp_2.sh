#! /bin/bash
#PBS -P gah51684
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:0:0
#PBS -j oe
#PBS -k oed
#PBS -o job_260212_cyp2.log

cd $PBS_O_WORKDIR


# ★これが最重要
module purge
module load python/3.12/3.12.9

# venv activate
source /home/aah18044co/github/XAI-DDI/boltz_env_py312/bin/activate

echo "=== env check ==="
which python
python -V
python -c "import sys; print(sys.executable)"
ldd "$(which python)" | grep libpython || true

which boltz

echo "=== job start ==="
python 260212_cyp_loop2.py