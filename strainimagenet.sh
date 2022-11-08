#!/bin/bash

#SBATCH --job-name=in_test
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=4g
#SBATCH --gres=gpu:3
#SBATCH --time=140:00:00
#SBATCH --account=precisionhealth_project1
#SBATCH --partition=gpu
#SBATCH --mail-user=achowdur+slurm@umich.edu
#SBATCH --export=ALL

# python job to run
# conda activate ldm
#python main.py --base models/first_stage_models/vq-f8-n256/config.yaml -r ../../experiment/vqmodel/logs/2022-09-21T01-20-11_config/ -l /nfs/turbo/umms-tocho/code/achowdur/experiment/vqmodel/logs -p bigsrhldm -t True --gpus 0,1,2,3
python main.py --base configs/latent-diffusion/cin-ldm-vq-f8.yaml -l /nfs/turbo/umms-tocho/code/achowdur/experiment/ldm/logs -p hybridcldm -t True --gpus 0,1,2

