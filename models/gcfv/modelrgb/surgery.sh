#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


python ../../../utils/surgery.py \
-f '/scratch/groups/lsdavis/yixi/face_ss/utils/VGG_ILSVRC_16_layers_deploy.prototxt' \
-c '/scratch/groups/lsdavis/yixi/face_ss/utils/VGG_ILSVRC_16_layers.caffemodel' \
-t 'deploy_modeldefault.prototxt' \
-o 'vgg16fc.caffemodel'


