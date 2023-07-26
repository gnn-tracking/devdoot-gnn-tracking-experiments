#!/usr/bin/env bash

#SBATCH --job-name=pl-run
#SBATCH --output=slurm_logs/pl-run-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-type=fail
#SBATCH --mail-user=dc3896@princeton.edu
#SBATCH --signal=SIGUSR1@90

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun \
  python3 run_graph_construction.py fit\
  --model configs/gc/model.yml \
  --trainer configs/gc/train.yml \
  --data configs/gc/data.yml \
  $@
