#!/bin/bash
#SBATCH --job-name=train_cond_encoder
#SBATCH --output=train_cond_encoder-%j.log
#SBATCH --error=train_cond_encoder-%j.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00:00

#echo 'export WANDB_API_KEY="94ad1ee0e14faa7ca831e3325dc339ada652c154"' >> ~/.bashrc
source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

echo "Starting script..."

python -m train_conditional_encoder --dataset_name='rocstories' --encoder_name='bert-base-cased' --project_name='pgwtd'
 
echo "Script finished."