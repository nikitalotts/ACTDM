#!/bin/bash
#SBATCH --job-name=make_statistics
#SBATCH --output=%j-make_statistics.log
#SBATCH --error=%j-make_statistics.log
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=0:15:00
#SBATCH --account=proj_1743

source ~/.bashrc
eval "$(conda shell.bash hook)"

export PYTHONPATH=/home/nklotts/tencdm-v3:$PYTHONPATH

module purge
module load Python

conda deactivate 
conda activate pgwtd

echo "Starting script..."

python3 -m data.make_statistics --dataset_name='rocstories' --encoder_name='bert-base-cased'

echo "Script finished"