#!/bin/bash
#SBATCH --job-name=calc_stat
#SBATCH --output=calc_statVVV-%j.log 
#SBATCH --error=calc_statVVV-%j.err   
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=5:00:00

#echo 'export WANDB_API_KEY="94ad1ee0e14faa7ca831e3325dc339ada652c154"' >> ~/.bashrc 
source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

export WANDB_MODE=offline

echo "Starting script..."

python test_gen.py --device-mode cpu
#python check_loads.py --device-mode cpu
 
echo "Script finished."