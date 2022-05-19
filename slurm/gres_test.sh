#!/bin/bash
#
# gres_test.bash
# Submit as follows:
# sbatch -p gpu --gres=gpu:4 -n4 gres_test.bash
#
echo JOB $SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
srun --gres=gpu:1 -n1 --exclusive show_device.sh &
srun --gres=gpu:1 -n1 --exclusive show_device.sh &
srun --gres=gpu:1 -n1 --exclusive show_device.sh &
srun --gres=gpu:1 -n1 --exclusive show_device.sh &
wait
