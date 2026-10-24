#!/bin/bash
#SBATCH --job-name=probe_geometry
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Оценка: сбор банка на 20000 векторов -- минуты, геометрия (cdist 512x20000,
# сортировка и локальный PCA) -- минуты, плюс до ~30 мин на загрузку.
# Итого ~40 мин; лимит 2 ч с запасом.
#SBATCH --time=02:00:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Probe: геометрия латентного пространства (hole ratio, r_tok, интерполяция)"

torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 probe_latent_geometry.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Probe finished."
