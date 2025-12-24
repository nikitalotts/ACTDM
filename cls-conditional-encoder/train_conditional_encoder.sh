#!/bin/bash
#SBATCH --job-name=calc_stat
#SBATCH --output=calc_stat-%j.log 
#SBATCH --error=calc_stat-%j.err   
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00

#echo 'export WANDB_API_KEY="94ad1ee0e14faa7ca831e3325dc339ada652c154"' >> ~/.bashrc
source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

echo "Starting script..."

python -m model.train_conditional_encoder --dataset_name='rocstories' --encoder_name='bert-base-cased' --project_name='pgwtd'
 
echo "Script finished."