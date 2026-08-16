#!/bin/bash
#SBATCH --job-name=eval_diffusion
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"
module purge
module load Python
conda deactivate
conda activate pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Starting diffusion evaluation (single run)..."

# порт из номера задания: стадия eval пускает 6 копий этого скрипта, на общей
# ноде фиксированный порт валит все, кроме первой
torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 eval_diffusion.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Diffusion evaluation finished."
