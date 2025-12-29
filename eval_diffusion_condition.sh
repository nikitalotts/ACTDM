#!/bin/bash
#SBATCH --job-name=eval_diffusion_conditional
#SBATCH --output=%j-eval_diffusion_conditional.log
#SBATCH --error=%j-eval_diffusion_conditional.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate
conda activate pgwtd

echo "Starting script..."

torchrun --master_port=31500 --nproc_per_node=1 eval_diffusion.py \
    --dataset_name rocstories \
    --scheduler sd \
    --coef_d 9 \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --mode transformer \
    --project_name='pgwtd' \
    --eval \
    --use_conditional_encoder \
    --is_conditional \
    --classifier_guidance_scale=1500.0

echo "Script finished."