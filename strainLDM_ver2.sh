#!/bin/bash

#SBATCH --job-name=ldm
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=4g
#SBATCH --gres=gpu:4
#SBATCH --time=140:00:00
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --mail-user=achowdur@umich.edu
#SBATCH --export=ALL

# python job to run
# conda activate ldm
# python main.py --base models/first_stage_models/vq-f8-n256/config.yaml -r ../../experiment/vqmodel/logs/2022-09-21T01-20-11_config/ -l /nfs/turbo/umms-tocho/code/achowdur/experiment/vqmodel/logs -p bigsrhldm -t True --gpus 0,1,2,3
python main.py --base configs/latent-diffusion/srh-ldm-vq-8.yaml -l /nfs/turbo/umms-tocho/code/achowdur/experiment/ldm/logs -p uncondldm -t True --gpus 0,1,2,3

