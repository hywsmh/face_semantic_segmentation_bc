#!/bin/bash

#SBATCH --time 10:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6


python ./solve.py
