#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


python ../../../utils/surgery_flow.py \
-f '/scratch/groups/lsdavis/yixi/face_ss/models/camvid/modelrgb/deploy.prototxt' \
-c '/scratch/groups/lsdavis/yixi/face_ss/models/camvid/modelrgb/snapshots_camvid500500/train_lr1e-10_12000_1e-9_8000_9000_1e-10/_iter_30000.caffemodel' \
-t 'deploy.prototxt' \
-o 'camvidrgb_surg.caffemodel'


