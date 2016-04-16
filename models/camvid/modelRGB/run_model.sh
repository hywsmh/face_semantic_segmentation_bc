#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

#source ./load_caffe_dependencies.sh

python ./solve.py > solve.out
