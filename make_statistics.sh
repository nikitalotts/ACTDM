#!/bin/bash
#SBATCH --job-name=make_statistics
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

source run_flags.sh

echo "Starting script..."

python3 -m data.make_statistics --encoder_name='bert-base-cased' ${DATA_FLAGS}

echo "Script finished"