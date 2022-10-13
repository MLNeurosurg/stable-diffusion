#!/bin/bash

#SBATCH --job-name=ldm
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=6g
#SBATCH --gres=gpu:3
#SBATCH --time=300:00:00
#SBATCH --account=precisionhealth_project1
#SBATCH --partition=gpu
#SBATCH --mail-user=achowdur@umich.edu
#SBATCH --export=ALL

# python job to run
# conda activate ldm
# python main.py --base models/first_stage_models/vq-f8-n256/config.yaml -r ../../experiment/vqmodel/logs/2022-09-21T01-20-11_config/ -l /nfs/turbo/umms-tocho/code/achowdur/experiment/vqmodel/logs -p bigsrhldm -t True --gpus 0,1,2,3
python main.py --base configs/latent-diffusion/SRH-ldm-vq-f8-condition.yaml -r ../../experiment/ldm/logs/2022-10-13T16-13-36_SRH-ldm-vq-f8-condition/ -l /nfs/turbo/umms-tocho/code/achowdur/experiment/ldm/logs -p condldm -t True --gpus 0,1,2,

