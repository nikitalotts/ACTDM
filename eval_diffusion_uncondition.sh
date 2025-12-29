#!/bin/bash
#SBATCH --job-name=eval_diffusion_unconditional
#SBATCH --output=%j-eval_diffusion_unconditional.log
#SBATCH --error=%j-eval_diffusion_unconditional.log
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

torchrun --nproc_per_node=1 eval_diffusion.py \
    --dataset_name rocstories \
    --scheduler sd \
    --coef_d 9 \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --mode transformer \
    --project_name='pgwtd' \
    --eval #  --use_conditional_encoder --is_conditional

echo "Script finished."