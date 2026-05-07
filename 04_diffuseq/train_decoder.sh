#!/bin/bash
#SBATCH --job-name=train_decoder
#SBATCH --output=%j-train_decoder-.log
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

export PYTHONPATH=/home/nklotts/tencdm-v3:$PYTHONPATH

module purge
module load Python

conda deactivate 
conda activate pgwtd

echo "Starting script..."

python -m model.train_decoder --dataset_name='rocstories' --encoder_name='bert-base-cased' --project_name='pgwtd'
 
echo "Script finished."