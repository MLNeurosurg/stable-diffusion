#!/bin/bash

#SBATCH --job-name=KL
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=4g
#SBATCH --gres=gpu:3
#SBATCH --time=120:00:00
#SBATCH --account=precisionhealth_project1
#SBATCH --partition=gpu
#SBATCH --mail-user=achowdur@umich.edu
#SBATCH --export=ALL

# python job to run
conda activate ldm
python main.py --base configs/autoencoder/autoencoder_kl_32x32x4.yaml -r ../../experiment/klmodel/logs/2022-09-20T00-24-56_autoencoder_kl_32x32x4/ -l /nfs/turbo/umms-tocho/code/achowdur/experiment/klmodel/logs -p autokl -t True --gpus 0,1,2

