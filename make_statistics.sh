#!/bin/bash
#SBATCH --job-name=Acalc_Stat
#SBATCH --output=Acalc_Stat-%j.log
#SBATCH --error=Acalc_Stat-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --account=proj_1743

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