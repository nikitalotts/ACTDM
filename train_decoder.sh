#!/bin/bash
#SBATCH --job-name=calc_stat
#SBATCH --output=calc_stat-%j.log 
#SBATCH --error=calc_stat-%j.err   
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

echo "Starting script..."

python -m model.train_decoder --dataset_name='rocstories' --encoder_name='bert-base-cased' --project_name='pgwtd'
 
echo "Script finished."