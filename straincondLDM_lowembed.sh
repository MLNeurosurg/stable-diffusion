#!/bin/bash

#SBATCH --job-name=cldmhbd
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=18g
#SBATCH --gres=gpu:4
#SBATCH --time=300:00:00
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --mail-user=achowdur+slurm@umich.edu
#SBATCH --export=ALL

# python job to run
# conda activate ldm
# python main.py --base models/first_stage_models/vq-f8-n256/config.yaml -r ../../experiment/vqmodel/logs/2022-09-21T01-20-11_config/ -l /nfs/turbo/umms-tocho/code/achowdur/experiment/vqmodel/logs -p bigsrhldm -t True --gpus 0,1,2,3
python main.py --base configs/latent-diffusion/SRH-ldm-vq-f8-condition_hybrid.yaml -r /nfs/turbo/umms-tocho/code/achowdur/experiment/ldm/logs/2022-10-25T17-12-44_SRH-ldm-vq-f8-condition_hybrid -l /nfs/turbo/umms-tocho/code/achowdur/experiment/ldm/logs -p hybridcldm -t True --gpus 0,1,2,3,

