#!/bin/bash
#SBATCH --mem 8000
#SBATCH -t 25:00:00
#SBATCH -N 1
#SBATCH -n 1

module load python-3.6.3

python3 conv2layers.py --name "2conv"
