#!/bin/bash
#SBATCH --job-name=eval_diffusion
#SBATCH --output=%j-eval_diffusion.log
#SBATCH --error=%j-eval_diffusion.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"
module purge
module load Python
conda deactivate
conda activate pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Starting diffusion evaluation (single run)..."

torchrun --master_port=31250 --nproc_per_node=1 eval_diffusion.py \
    --dataset_name rocstories \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Diffusion evaluation finished."
