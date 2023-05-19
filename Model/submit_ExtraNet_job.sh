#!/bin/sh
#SBATCH -c 12
#SBATCH --gres=gpu:1 -p g24

# python inference.py
python train.py