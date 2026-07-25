#!/bin/bash
#SBATCH --job-name=train_cond_encoder
#SBATCH --output=train_cond_encoder-%j.log
#SBATCH --error=train_cond_encoder-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

source mode_flags.sh

echo "Starting script..."

python -m train_conditional_encoder_augmented --dataset_name='rocstories' --encoder_name='bert-base-cased' --project_name='actdm' ${DATA_FLAGS}
 
echo "Script finished."