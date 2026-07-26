#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --output=%j-train_diffusion.log
#SBATCH --error=%j-train_diffusion.log
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=57:00:00


source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Starting script..."

torchrun --nproc_per_node=4 train_diffusion.py --encoder_name bert-base-cased --project_name='actdm' ${ARCH_FLAGS} ${DATA_FLAGS}

echo "Script finished."