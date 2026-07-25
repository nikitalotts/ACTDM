#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --output=train_diffusion-%j.log
#SBATCH --error=train_diffusion-%j.log
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=11:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

export WANDB_MODE=offline

source mode_flags.sh

echo "Starting script..."

torchrun --master_port=31503 --nproc_per_node=4 train_diffusion.py \
    --dataset_name rocstories \
    --scheduler sd \
    --coef_d 9 \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --mode transformer \
    --project_name='actdm' \
    ${MODE_FLAGS} ${DATA_FLAGS}
 
echo "Script finished."