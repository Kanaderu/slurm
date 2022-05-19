#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH --job-name=test
#SBATCH --time=2-00:00
#SBATCH --mem=36000
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
# the -u option means 'unbuffered',
# which should continuously write output to the .out file
srun --gres=gpu:1 -u /home/fandavid/.conda/envs/tf2/bin/python /home/fandavid/test/tf_test.py
##SBATCH --output=/storage/test/test.out
##SBATCH --error=/storage/test/test.err
