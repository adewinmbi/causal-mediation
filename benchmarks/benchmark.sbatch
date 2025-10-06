#!/bin/bash
#SBATCH -N 1               
#SBATCH -n 4              
#SBATCH --gres=gpu:h200:1 
#SBATCH -p mit_preemptable 
#SBATCH -t 02:00:00
#SBATCH --mem=128G

module load miniforge
source activate chimera
uv sync
source ../.venv/bin/activate
python benchmark.py