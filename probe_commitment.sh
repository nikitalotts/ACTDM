#!/bin/bash
#SBATCH --job-name=probe_commit
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Оценка: 4 батча x 315 шагов солвера (базовая траектория 50 плюс хвосты из
# 10 точек ветвления) -- около 10 мин счета, плюс до ~30 мин на загрузку.
# Итого ~45 мин; у genie шаг дороже, отсюда лимит 4 ч с запасом.
# Результат досохраняется после КАЖДОГО батча, так что срез по лимиту теряет
# максимум один батч, а не весь прогон.
#SBATCH --time=04:00:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Probe: кривая предрешенности (ветвление обратного процесса)"

torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 probe_commitment.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Probe finished."
