#!/bin/bash
#SBATCH --job-name=eval_diffusion
#SBATCH --output=eval_diffusion-%j.log
#SBATCH --error=eval_diffusion-%j.err
#SBATCH --cpus-per-task=12
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

export WANDB_MODE=offline

echo "Starting eval..."

# Параметры из названия чекпоинта:
# tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0
# rocstory-bert-base-cased-sd-9-spt_100000.pth
#
# tencdm -> emb=False (НЕ передаем --emb)
# scheduler=sd, coef_d=9
# dataset=rocstories
# swap_cfg_coef=0.0

torchrun --nproc_per_node=1 eval_diffusion.py \
    --dataset_name rocstories \
    --scheduler sd \
    --coef_d 9 \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --mode transformer \
    --project_name eval_test

#torchrun --nproc_per_node=1 eval_diffusion.py

echo "Eval finished."